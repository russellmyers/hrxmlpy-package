import pandas as pd
import numpy as np
import random
import math
import collections


class EnhancedIsolationTree:
    """
    Used within EnhancedIsolationForest class.
    represents a single tree within the forest
    """

    def __init__(self):
        self.nodes = None

    def choose_feature(self, df):
        feature_chosen = False
        features_avail = [i for i in range(df.shape[1])]

        while len(features_avail) > 0:
            r = random.randint(0, len(features_avail) - 1)
            col_num = features_avail[r]
            col = df.columns[col_num]
            if col_num in self.cat_column_nums:
                num_unique = df[col].nunique()
                if num_unique <= 1:
                    features_avail.remove(col_num)
                else:
                    return col, col_num
            else:
                min, max = self.get_min_max(df, col)
                if min == max:
                    features_avail.remove(col_num)
                else:
                    return col, col_num

        return None, None

    def get_min_max(self, df, col):
        max = df[col].max()
        min = df[col].min()
        return min, max

    def create_node(
        self, parent, branch, feat, spl_point, depth, num_recs, left_child=None, right_child=None, feature_num=None
    ):

        return {
            "parent": parent,
            "branch": branch,
            "feat": feat,
            "feat_num": feature_num,
            "spl_point": spl_point,
            "depth": depth,
            "num_recs": num_recs,
            "left_child": left_child,
            "right_child": right_child,
        }

    def add_node(
        self, nodes, parent, branch, feat, spl_point, depth, num_recs, left_child=-1, right_child=-1, feature_num=None
    ):

        if len(nodes) == 0:
            nodes["left"] = []
            nodes["right"] = []
            nodes["feat"] = []
            nodes["thresh"] = []
            nodes["depth"] = []
            nodes["num_samples"] = []

        nodes["left"].append(left_child)
        nodes["right"].append(right_child)
        nodes["feat"].append(-2 if feature_num is None else feature_num)
        nodes["thresh"].append(-2 if spl_point is None else spl_point)
        nodes["depth"].append(depth)
        nodes["num_samples"].append(num_recs)

        new_node_num = len(nodes["left"]) - 1

        if parent is None:
            pass
        else:
            if branch == "L":
                nodes["left"][parent] = new_node_num
            else:
                nodes["right"][parent] = new_node_num

        return new_node_num  # index of last added

    def build(
        self, df, curr_depth=0, curr_parent_num=None, curr_branch=None, nodes={}, max_depth=None, num_features=None
    ):

        if curr_depth == 0:
            num_features = df.shape[1]
            max_depth = math.ceil(math.log(df.shape[0], 2))
            # print('max depth: ',max_depth)

        if df.shape[0] <= 1:
            # leaf node
            node_num = self.add_node(nodes, curr_parent_num, curr_branch, None, None, curr_depth, df.shape[0])
            return nodes

        if curr_depth >= max_depth:
            node_num = self.add_node(nodes, curr_parent_num, curr_branch, None, None, curr_depth, df.shape[0])
            return nodes

        f, f_num = self.choose_feature(df)
        if f is None:
            spl_point = None  # No feature chosen - all vals same
        else:
            if f_num in self.cat_column_nums:
                # print('cat col: ',f)
                all_cat_vals = df.iloc[:, f_num].unique()
                # print(all_cat_vals)
                r = random.randint(0, len(all_cat_vals) - 1)
                spl_point = all_cat_vals[r]
                # print('r',r,'spl_point',spl_point)

            else:
                min, max = self.get_min_max(df, f)
                if min == max:
                    spl_point = min
                    print("all equal: ", f, df)
                else:
                    spl_point = random.uniform(min, max - 0.000001)
        num_recs = df.shape[0]

        curr_node_num = self.add_node(
            nodes, curr_parent_num, curr_branch, f, spl_point, curr_depth, num_recs, feature_num=f_num
        )
        # nodes.append(curr_node)
        # curr_node_num = len(nodes) - 1

        if f is None:  # No feature chosen - all vals same
            return nodes

        left_samples = df[df[f] == spl_point] if f_num in self.cat_column_nums else df[df[f] <= spl_point]
        right_samples = df[~(df[f] == spl_point)] if f_num in self.cat_column_nums else df[df[f] > spl_point]

        if f_num in self.cat_column_nums:
            # print('left',left_samples[:3])
            # print('right',right_samples[:3])
            pass

        # Left
        nodes = self.build(left_samples, curr_depth + 1, curr_node_num, "L", nodes, max_depth, num_features)
        # Right
        nodes = self.build(right_samples, curr_depth + 1, curr_node_num, "R", nodes, max_depth, num_features)

        return nodes

    def build_tree(self, df, cat_columns, cat_column_nums):
        self.cat_columns = cat_columns
        self.cat_column_nums = cat_column_nums

        self.nodes = None
        self.nodes = self.build(
            df, curr_depth=0, curr_parent_num=None, nodes={}, curr_branch=None, max_depth=None, num_features=None
        )

    def traverse_tree_example(self, sample, nodes_used=[], curr_node_num=0):

        # node = self.nodes[curr_node_num]
        if self.nodes["feat"][curr_node_num] == -2:
            # if node['feat'] is None:
            nodes_used.append(curr_node_num)
            return nodes_used
        else:
            df_val = sample[self.nodes["feat"][curr_node_num]]
            nodes_used.append(curr_node_num)
            if self.nodes["feat"][curr_node_num] in self.cat_column_nums:
                if df_val == self.nodes["thresh"][curr_node_num]:
                    child_num = self.find_child(curr_node_num, "L")
                else:
                    child_num = self.find_child(curr_node_num, "R")
            else:
                if df_val <= self.nodes["thresh"][curr_node_num]:
                    child_num = self.find_child(curr_node_num, "L")
                else:
                    child_num = self.find_child(curr_node_num, "R")
            return self.traverse_tree_example(sample, nodes_used, child_num)

    def traverse_tree(self, df, add_depths_for_leaf_node=True):

        df_ar = df.values
        sample_ptrs = []
        node_indicators = []

        depths_per_sample = []
        feats_used_per_sample = []
        last_feats_used_per_sample = []
        leaf_sample_size_per_sample = []

        for i in range(0, df.shape[0]):
            sample_ptr = len(node_indicators)
            sample_ptrs.append(sample_ptr)
            nodes_used = self.traverse_tree_example(df_ar[i], nodes_used=[], curr_node_num=0)
            depth = len(nodes_used) - 1
            if add_depths_for_leaf_node:
                num_samples_at_leaf = self.nodes["num_samples"][nodes_used[-1]]
                depth += EnhancedIsolationForest.average_path_length([num_samples_at_leaf])[0]
            depths_per_sample.append(depth)
            node_indicators.extend(nodes_used)
            feats_used = [self.nodes["feat"][node_num] for node_num in nodes_used[:-1]]
            feats_used_per_sample.append(feats_used)
            if len(feats_used) == 0:
                pass
            else:
                last_feats_used_per_sample.append(feats_used[-1])
            leaf_sample_size_per_sample.append(self.nodes["num_samples"][nodes_used[-1]])
            # print(i,depth,feat_used)
            # depths.append(depth)
            # features_used.append(feat_used)
            # num_recs_leaf_list.append(num_recs_leaf)
        return (
            sample_ptrs,
            node_indicators,
            depths_per_sample,
            feats_used_per_sample,
            last_feats_used_per_sample,
            leaf_sample_size_per_sample,
        )

    def find_child(self, curr_node_num, branch):

        if branch == "L":
            return self.nodes["left"][curr_node_num]
        else:
            return self.nodes["right"][curr_node_num]


class EnhancedIsolationForest:
    """
    Enhanced Isolaton Forest Class.

    Extends functionality of scikit-learn Isolation Forest by catering for categorical features.

    parameters:
      - df_f: pandas dataframe with features only (no labels or id columns)
      - num_trees: default 100
      - verbose: default 0

     max depth defaults to log2 of number of examples
    """

    def __init__(self, num_trees=200, max_samples="auto", verbose=0, random_state=42):
        self.trees = []
        self.verbose = verbose
        self.num_trees = num_trees
        self.max_samples = max_samples
        self.use_samples = None  # set in fit() method
        self.forest_depths = None
        self.all_features_used = None
        self.random_state = 42

    def set_params(self, **params):
        self.trees = params.get("trees", [])
        self.verbose = params.get("verbose", 0)
        self.num_trees = params.get("num_trees", 200)
        self.max_samples = params.get("max_samples", "auto")
        self.use_samples = params.get("use_samples", None)
        self.forest_depths = params.get("forest_depths", None)
        self.all_features_used = params.get("all_features_used", None)
        self.random_state = params.get("random_state", 42)
        return self

    def get_params(self):
        return dict(
            trees=self.trees,
            verbose=self.verbose,
            num_trees=self.num_trees,
            max_samples=self.max_samples,
            use_samples=self.use_samples,
            forest_depths=self.forest_depths,
            all_features_used=self.all_features_used,
        )

    def fit(self, df_f, verbose=0):
        auto_samples = 256
        random.seed(self.random_state)
        self.cat_columns = list(df_f.select_dtypes(include="object").columns)
        print("Categorical: ", self.cat_columns)
        self.cat_column_nums = [df_f.columns.get_loc(c) for c in self.cat_columns]
        print("Categorical column nums: ", self.cat_column_nums)

        self.trees = [None for i in range(self.num_trees)]

        if self.max_samples == "auto":
            if df_f.shape[0] > auto_samples:
                self.use_samples = auto_samples
            else:
                self.use_samples = df_f.shape[0]
        else:
            self.use_samples = self.max_samples

        for i in range(self.num_trees):
            if i % 10 == 0:
                print("training tree: ", i, "/", self.num_trees)
            self.trees[i] = EnhancedIsolationTree()

            self.trees[i].build_tree(df_f.sample(self.use_samples), self.cat_columns, self.cat_column_nums)

    def add_scores(self, df, avg_depths):
        df_with_scores = df.copy()
        df_with_scores["score"] = avg_depths
        return df_with_scores

    def sort_and_rank(self, df):
        df_sorted = df.copy()
        df_sorted = df_sorted.sort_values(by="score")
        df_sorted["rank"] = range(1, df_sorted.shape[0] + 1)
        return df_sorted

    def predict(self, df, thresh=3):
        df_new = df.copy()
        df_desc = df["score"].describe()
        df_new["predict"] = df_new["score"].apply(lambda x: 1 if x < df_desc["mean"] - thresh * (df_desc["std"]) else 0)
        return df_new

    @staticmethod
    def average_path_length(num_samples):
        # as per isolation forest original paper

        eulers_constant = 0.5772156649

        av_lengths = []

        for num_samples_leaf in num_samples:
            if num_samples_leaf == 0 or num_samples_leaf == 1:
                av_lengths.append(0)
            else:
                av_lengths.append(
                    2 * ((np.log(num_samples_leaf - 1) + eulers_constant))
                    - 2 * (num_samples_leaf - 1) / num_samples_leaf
                )

        return av_lengths

    def determine_important_features(
        self, df_X_scaled, df, num_top_features, show_progress_after_each=20, show_diagnostics_flag=False
    ):

        all_trees_feats_used_per_sample = []
        all_trees_last_feats_used_per_sample = []

        for i in range(0, len(self.trees)):
            if i % show_progress_after_each == 0:
                print("traversing tree: ", i, "/", len(self.trees))
            (
                sample_ptrs,
                node_indicators,
                depths_per_sample,
                feats_used_per_sample,
                last_feats_used_per_sample,
                leaf_sample_size_per_sample,
            ) = self.trees[i].traverse_tree(df_X_scaled)
            all_trees_feats_used_per_sample.append(feats_used_per_sample)
            all_trees_last_feats_used_per_sample.append(last_feats_used_per_sample)

        all_trees_feats_used_per_sample_np = np.array(all_trees_feats_used_per_sample)

        most_used_per_sample = []
        last_used_per_sample = []

        for i in range(all_trees_feats_used_per_sample_np.shape[1]):
            sample_feats_used = all_trees_feats_used_per_sample_np[:, i]
            list_list = list(sample_feats_used)
            flat_list = [item for sublist in list_list for item in sublist]
            counter_most_used = collections.Counter(flat_list)
            counter_most_used_in_order = counter_most_used.most_common((3))
            most_used_per_sample.append(counter_most_used_in_order)

        all_trees_last_feats_used_per_sample_np = np.array(all_trees_last_feats_used_per_sample)
        for i in range(all_trees_last_feats_used_per_sample_np.shape[1]):
            sample_last_feats_used = all_trees_last_feats_used_per_sample_np[:, i]
            counter_last_used = collections.Counter(sample_last_feats_used)
            counter_last_used_in_order = counter_last_used.most_common((3))
            last_used_per_sample.append(counter_last_used_in_order)

        cols = df_X_scaled.columns
        most_used_per_sample_feat_nums = [[cols[x[0]] for x in tup] for tup in most_used_per_sample]
        last_used_per_sample_feat_nums = [[cols[x[0]] for x in ddd] for ddd in last_used_per_sample]

        most_used_np = np.array(most_used_per_sample_feat_nums)
        last_used_np = np.array(last_used_per_sample_feat_nums)
        df_with_feature_importances = df.copy()
        for i in range(0, num_top_features):
            if most_used_np.shape[1] < i + 1:
                df_with_feature_importances["F" + str(i + 1)] = cols[
                    0
                ]  # arbitrarily use first feature.. Not enough top features
            else:
                df_with_feature_importances["F" + str(i + 1)] = most_used_np[:, i]

        for i in range(0, num_top_features):
            if last_used_np.shape[0] < i + 1:
                df_with_feature_importances["LF" + str(i + 1)] = cols[
                    0
                ]  # arbitrarily use first feature.. Not enough top features
            else:
                df_with_feature_importances["LF" + str(i + 1)] = last_used_np[:, i]

        return df_with_feature_importances

    def decision_function(self, df, add_depths_for_leaf_nodes=True):
        """
        Calc anomaly scores. (Also determines important features at same time - not done by scikit learn decision function)
        """
        if self.verbose >= 1:
            report_every = 1
        else:
            report_every = 10

        all_sample_ptrs = []
        all_node_indicators = []
        all_trees_depths_per_sample = []
        all_trees_feats_used_per_sample = []
        all_trees_last_feats_used_per_sample = []

        for i in range(0, len(self.trees)):
            if i % report_every == 0:
                print("scoring tree: ", i, "/", len(self.trees))
            (
                sample_ptrs,
                node_indicators,
                depths_per_sample,
                feats_used_per_sample,
                last_feats_used_per_sample,
                leaf_sample_size_per_sample,
            ) = self.trees[i].traverse_tree(df)
            all_trees_depths_per_sample.append(depths_per_sample)
            all_trees_feats_used_per_sample.append(feats_used_per_sample)
            all_trees_last_feats_used_per_sample.append(last_feats_used_per_sample)
            all_sample_ptrs.append(sample_ptrs)
            all_node_indicators.append(node_indicators)

        all_trees_feats_used_per_sample_np = np.array(all_trees_feats_used_per_sample)

        forest_depths = np.array(all_trees_depths_per_sample)
        avg_depths = np.mean(forest_depths, axis=0)
        avg_depths[avg_depths == 0] = 1  # cater for zero avg depth (can occur with very small sample sizes)
        avg_depths = 2 ** (-avg_depths / self.average_path_length([self.use_samples])[0])
        avg_depths = 0.5 - avg_depths

        return avg_depths  # , most_used_per_sample_feat_nums, last_used_per_sample_feat_nums
