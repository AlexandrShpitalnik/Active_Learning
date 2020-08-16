import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from tqdm import trange
from multiprocessing.pool import ThreadPool


class HiddenFunc:
    def __init__(self):
        pass

    @staticmethod
    def get_value(x):
        return x[1] * x[6] + x[8] / x[9] * np.sqrt(x[6] / x[7]) + np.pi * \
            np.sqrt(x[2]) + 1 / np.sin(x[3]) + np.log(x[2] + x[4])


# todo add iterm stats
class ActiveLearner:
    def __init__(self, weak_estimator_class, max_depth, mark_obj_func, n_estimators, n_jobs=4):
        self.weak_estimator_class = weak_estimator_class
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.mark_obj_func = mark_obj_func
        self.n_jobs = n_jobs
        self.ensemble = None
        self.use_clusters = None
        self.n_clusters = None
        self.cluster_model = None

        self.visualization_flag = False
        self.visualization_model = None
        self.freq = None
        self.metric_func = None
        self.test_x = None
        self.test_y = None
        self.history = None

    def add_visualization(self, test_x, test_y, freq, metric_func):
        self.visualization_flag = True
        self.test_x = test_x
        self.test_y = test_y
        self.freq = freq
        self.metric_func = metric_func
        self.visualization_model = XGBRegressor()

    def get_history(self):
        return self.history

    def __get_new_points_step(self, n_cnd_per_iter, batch_size):
        new_points_x = np.random.random_sample((n_cnd_per_iter, 10)) * 10

        if self.use_clusters:
            clust_labels = self.cluster_model.fit_predict(new_points_x)
        else:
            self.n_clusters = 1
            clust_labels = np.zeros(len(new_points_x))

        n_to_label_for_each_class = int(np.floor(batch_size / self.n_clusters))
        res_x = np.empty((0, 10))
        res_y = np.empty((0,))
        for cur_clust in range(self.n_clusters):
            clust_points = new_points_x[clust_labels == cur_clust]
            pred_y = [model.predict(clust_points) for model in self.ensemble]
            pred_y = np.array(pred_y).T
            disp_bottom_top = pred_y.std(axis=1).argsort()
            n_to_label = min(len(disp_bottom_top), n_to_label_for_each_class)
            idx_to_label = disp_bottom_top[-n_to_label:]
            to_add_x = clust_points[idx_to_label]
            to_add_y = np.apply_along_axis(self.mark_obj_func, 1, to_add_x)
            res_x, res_y = np.vstack((res_x, to_add_x)), np.hstack((res_y, to_add_y))
        return res_x, res_y

    def __fit_ensemble(self, x, y):
        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]
        set_size = int(np.floor(len(y) / self.n_estimators))
        start, stop = 0, set_size
        ds_parts = []

        for i in range(self.n_estimators):
            ds_parts.append((x[start:stop], y[start:stop]))
            start += set_size
            stop = stop + set_size if i < self.n_estimators - 2 else -1

        pool = ThreadPool(self.n_jobs)
        self.ensemble = pool.map(self.__fit_est_parallel, ds_parts)
        pool.close()
        pool.join()

    def __fit_est_parallel(self, ds_part):
        new_model = self.weak_estimator_class(max_depth=self.max_depth)
        new_model.fit(ds_part[0], ds_part[1])
        return new_model

    def expand_random(self, x_start, y_start, n_new_points, n_cnd_per_iter, batch_size, use_clust=True, n_clusters=100):
        self.use_clusters = use_clust
        self.n_clusters = n_clusters
        self.history = []
        if use_clust:
            self.cluster_model = KMeans(n_clusters=self.n_clusters, n_init=1)
        self.ensemble = None
        x, y = x_start, y_start
        n_iters = int(np.floor(n_new_points/batch_size))

        for i in trange(n_iters):

            self.__fit_ensemble(x, y)
            x_new, y_new = self.__get_new_points_step(batch_size=batch_size, n_cnd_per_iter=n_cnd_per_iter)
            x, y = np.vstack((x, x_new)), np.hstack((y, y_new))

            if self.visualization_flag and i > 0 and i % self.freq == 0:
                self.visualization_model.fit(x, y)
                y_pred = self.visualization_model.predict(self.test_x)
                loss = self.metric_func(self.test_y, y_pred)
                self.history.append(loss)
        return x, y


# todo add iterm stats
class ActiveLearnTest:
    def __init__(self, func=None, al_model_class=None, n_est=6, max_depth=4, n_jobs=4, visualisation_flag=False,
                 freq=50):
        self.hidden_func = func if func else HiddenFunc()
        al_model_class = al_model_class if al_model_class else ActiveLearner
        self.al_model = al_model_class(mark_obj_func=self.hidden_func.get_value, n_estimators=n_est,
                                       weak_estimator_class=DecisionTreeRegressor, max_depth=max_depth, n_jobs=n_jobs)
        self.loss = self.__mse
        self.visualisation_flag = visualisation_flag
        self.visualization_model = XGBRegressor()
        self.freq = freq
        self.history = None
        self.datasets = None

    @staticmethod
    def __mse(a, b):
        return np.sum((a-b)**2)/len(a)

    @staticmethod
    def __make_grid(start, stop, points_per_dim, n_dims):
        """
        make square grid
        """
        dim_interval = stop - start
        between_points = dim_interval/(points_per_dim - 1)
        single_dim = []
        for i in range(int(points_per_dim)):
            point = start + i * between_points
            single_dim.append(point)
        grid_tuples = pd.MultiIndex.from_product([single_dim for _ in range(n_dims)]).values
        grid = np.array([np.array(t) for t in grid_tuples])
        return grid

    def __get_points_sets(self, start_size, n_cnd_per_iter, batch_size, n_points, random_init, use_clust, n_clusters,
                          test_size):
        """
        n_cnd_per_iter: True - use random train points; False - use grid
        """
        if random_init:
            start_x = np.random.random_sample((start_size, 10))*10
        else:
            start_x = self.__make_grid(0.0000001, 10, 3, 10)  # 59049 points (3)
        start_y = np.apply_along_axis(self.hidden_func.get_value, 1, start_x)

        test_x = np.random.random_sample((test_size, 10)) * 10
        test_y = np.apply_along_axis(self.hidden_func.get_value, 1, test_x)

        if self.visualisation_flag:
            self.al_model.add_visualization(test_x, test_y, self.freq, self.loss)

        train_x_al, train_y_al = self.al_model.expand_random(start_x, start_y, batch_size=batch_size,
                                                             n_cnd_per_iter=n_cnd_per_iter, n_new_points=n_points,
                                                             use_clust=use_clust, n_clusters=n_clusters)
        add_size = len(train_y_al) - len(start_y)

        if self.visualisation_flag:
            self.history = []
            train_x_random, train_y_random = start_x, start_y
            n_steps = int(np.floor(add_size/batch_size))
            for i in trange(n_steps):
                step_add_x = np.random.random_sample((batch_size, 10)) * 10
                step_add_y = np.apply_along_axis(self.hidden_func.get_value, 1, step_add_x)
                train_x_random, train_y_random = np.vstack((train_x_random, step_add_x)), \
                    np.hstack((train_y_random, step_add_y))
                if i > 0 and i % self.freq == 0:
                    self.visualization_model.fit(train_x_random, train_y_random)
                    step_pred_y = self.visualization_model.predict(test_x)
                    step_loss = self.loss(step_pred_y, test_y)
                    self.history.append(step_loss)
        else:
            add_x = np.random.random_sample((add_size, 10)) * 10
            add_y = np.apply_along_axis(self.hidden_func.get_value, 1, add_x)
            train_x_random, train_y_random = np.vstack((start_x, add_x)), np.hstack((start_y, add_y))

        grid_points_per_dim = int(np.floor(pow(len(train_x_al), 1/10)))
        train_x_grid = self.__make_grid(0.0000001, 10, grid_points_per_dim, 10)
        train_y_grid = np.apply_along_axis(self.hidden_func.get_value, 1, train_x_grid)

        return [(train_x_al, train_y_al), (train_x_grid, train_y_grid), (train_x_random, train_y_random),
                (test_x, test_y)]

    def get_stats(self, start_size=60000, n_new_points=1050000, batch_size=800, n_cnd_per_iter=40000,
                  test_size=500000, use_clust=True, n_clusters=100, random_init=False):
        model = XGBRegressor()
        self.datasets = None
        al_set, grid_set, random_set, test_set = self.__get_points_sets(start_size=start_size,
                                                                        n_cnd_per_iter=n_cnd_per_iter,
                                                                        batch_size=batch_size, n_points=n_new_points,
                                                                        random_init=random_init, use_clust=use_clust,
                                                                        n_clusters=n_clusters, test_size=test_size)
        print(len(grid_set[1]))
        self.datasets = [al_set, grid_set, random_set, test_set]

        model.fit(al_set[0], al_set[1])
        al_pred_y = model.predict(test_set[0])
        al_mse = self.loss(al_pred_y, test_set[1])

        model.fit(grid_set[0], grid_set[1])
        grid_pred_y = model.predict(test_set[0])
        grid_mse = self.loss(grid_pred_y, test_set[1])

        model.fit(random_set[0], random_set[1])
        random_pred_y = model.predict(test_set[0])
        random_mse = self.loss(random_pred_y, test_set[1])

        return al_mse, grid_mse, random_mse

    def get_history(self):
        return self.history, self.al_model.get_history()
