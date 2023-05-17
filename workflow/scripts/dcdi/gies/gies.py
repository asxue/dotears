"""GIES algorithm.

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from cdt.causality.graph.model import GraphModel
from pandas import DataFrame, read_csv
from cdt.utils.Settings import SETTINGS
from cdt.utils.R import RPackages, launch_R_script
import numpy as np
import torch


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class GIES(GraphModel):
    """GIES algorithm **[R model]**.

    **Description:** Greedy Interventional Equivalence Search algorithm.
    A score-based Bayesian algorithm that searches heuristically the graph which minimizes
    a likelihood score on the data. The main difference with GES is that it
    accepts interventional data for its inference.

    **Required R packages**: pcalg

    **Data Type:** Continuous (``score='obs'``) or Categorical (``score='int'``)

    **Assumptions:** The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    Args:
        score (str): Sets the score used by GIES.
        verbose (bool): Defaults to ``cdt.SETTINGS.verbose``.

    Available scores:
        + int: GaussL0penIntScore
        + obs: GaussL0penObsScore

    .. note::
       Ref:
       D.M. Chickering (2002).  Optimal structure identification with greedy search.
       Journal of Machine Learning Research 3 , 507–554

       A. Hauser and P. Bühlmann (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs.
       Journal of Machine Learning Research 13, 2409–2464.

       P. Nandy, A. Hauser and M. Maathuis (2015). Understanding consistency in
       hybrid causal structure learning.
       arXiv preprint 1507.02608

       P. Spirtes, C.N. Glymour, and R. Scheines (2000).
       Causation, Prediction, and Search, MIT Press, Cambridge (MA)

    Example:
        >>> import networkx as nx
        >>> from cdt.causality.graph import GIES
        >>> from cdt.data import load_dataset
        >>> data, graph = load_dataset("sachs")
        >>> obj = GIES()
        >>> #The predict() method works without a graph, or with a
        >>> #directed or undirected graph provided as an input
        >>> output = obj.predict(data)    #No graph provided as an argument
        >>>
        >>> output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
        >>>
        >>> output = obj.predict(data, graph)  #With a directed graph
        >>>
        >>> #To view the graph created, run the below commands:
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """

    def __init__(self, score='obs', verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(GIES, self).__init__()
        self.scores = {'int': 'GaussL0penIntScore',
                       'obs': 'GaussL0penObsScore'}
        self.arguments = {'{FOLDER}': '/tmp/cdt_gies/',
                          '{FILE}': 'data.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': 'fixedgaps.csv',
                          '{TARGETS}': 'targets.csv',
                          '{SCORE}': 'GaussL0penIntScore',
                          '{VERBOSE}': 'FALSE',
                          '{LAMBDA}': '1',
                          '{OUTPUT}': 'result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.score = score

    def orient_undirected_graph(self, data, graph, targets=None):
        """Run GIES on an undirected graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.Graph): Skeleton of the graph to orient
            targets (): Skeleton of the graph to orient TODO

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        self.arguments['{SCORE}'] = self.scores[self.score]

        fe = DataFrame(nx.adj_matrix(graph, weight=None).todense())
        fg = DataFrame(1 - fe.values)

        results = self._run_gies(data, fixedGaps=fg, targets=targets, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph, targets=None):
        """Run GIES on a directed_graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient
            targets (): Skeleton of the graph to orient TODO

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.

        """
        warnings.warn("GIES is ran on the skeleton of the given graph.")
        return self.orient_undirected_graph(data, nx.Graph(graph))

    def create_graph_from_data(self, data):
        """Run the GIES algorithm.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.
        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[self.score]
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        results = self._run_gies(data, targets=targets, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_gies(self, data, fixedGaps=None, targets=None, lambda_gies=None, verbose=True):
        """Setting up and running GIES with all arguments."""
        # Run gies
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_gies' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_gies' + id + '/'
        self.arguments['{LAMBDA}'] = str(lambda_gies)

        def retrieve_result():
            return read_csv('/tmp/cdt_gies' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_gies' + id + '/data.csv', header=False, index=False)
            if fixedGaps is not None:
                fixedGaps.to_csv('/tmp/cdt_gies' + id + '/fixedgaps.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            if targets is not None:
                # to make it work with R:
                # if there is only one column, add an empty one
                if targets.shape[1] == 1:
                    targets['dummy'] = np.nan
                targets.to_csv('/tmp/cdt_gies' + id + '/targets.csv', index=False, header=False)
                self.arguments['{INTERVENTION}'] = 'TRUE'
            else:
                self.arguments['{INTERVENTION}'] = 'FALSE'


            gies_result = launch_R_script("{}/gies.R".format(os.path.dirname(os.path.realpath(__file__))),
                                          self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_gies' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_gies' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_gies' + id + '')
        return gies_result


class LinearMasked(torch.nn.Module):
    def __init__(self, adj):
        super(LinearMasked, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(np.zeros_like(adj)))
        self.adj = torch.Tensor(adj)

    def forward(self, data, mask):
        out = torch.matmul(data, self.adj * self.weights)
        return torch.mean(torch.sum(0.5 * ((data - out) * mask)**2, dim=0) / torch.sum(mask, dim=0))

def retrain(adj, train_data, valid_data, max_iter, batch_size):

    model = LinearMasked(adj)
    optimizer = torch.optim.Adagrad(model.parameters())
    best_valid_score = -np.inf
    full_patience = 100
    flag_max_iter = True

    for iter in range(max_iter):
        x, mask, _ = train_data.sample(batch_size)

        loss = model(x, mask)
        train_score = -loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if valid_data is not None:
            with torch.no_grad():
                x, mask, _ = valid_data.sample(valid_data.num_samples)
                valid_score = -model(x, mask).item()
        else:
            valid_score = train_score

        if valid_score > best_valid_score + 1e-4:
            best_valid_score = valid_score
            # compute best model training score
            x, mask, _ = train_data.sample(train_data.num_samples)
            best_train_score = -model(x, mask).item()
            #restore patience
            patience = full_patience
        else:
            patience -= 1

        if iter % 100 == 0:
            print("Iteration: {}, score_train: {:.5f} , score_valid : {:.5f}, best_train_score : {:.5f}. best_valid_score {:.5f}, patience: {}".format(iter, train_score, valid_score, best_train_score, best_valid_score, patience))
        if patience == 0:
            flag_max_iter = False
            break

    return best_train_score, best_valid_score, flag_max_iter
