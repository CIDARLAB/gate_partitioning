import copy
import math
import random
from typing import (
    List,
)
from numba import jit

import matplotlib.pyplot as plt
import metis
import networkx as nx


#########################################
# Circuit partition
#########################################

def run_metis(input_graph, number_of_parts):
    # this function partitions a network into k subgraphs
    partition = metis.part_graph(
        input_graph,
        nparts=number_of_parts,
        recursive=True,
    )
    return partition


def find_best_cut(input_graph, nonprimitives):
    """ for a given G, obtain the optimal nparts that gives the minimum cuts """
    # remove input and output nodes
    graph_copy = copy.deepcopy(input_graph)
    [graph_copy.remove_node(node) for node in nonprimitives]

    min_nodes = 5
    # The five seems arbitrary.
    possible_partitions = list(range(2, int(len(graph_copy.nodes()) / min_nodes) + 1))

    # find n (block number) that gives the minimum cuts
    minimum_cuts = float('inf')
    optimized_partition = None
    for n in possible_partitions:
        objective_value, parts = run_metis(graph_copy, n)
        if objective_value < minimum_cuts:
            optimized_partition = objective_value, parts
            minimum_cuts = objective_value

    return optimized_partition


#########################################
# Gate assignment
#########################################

def load_gate_info(inputfile):
    lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
    gateLib = {}
    header = lines[0].split('\t')
    for line in lines[1:]:
        tokens = line.split('\t')
        gate = tokens[0]
        gateLib[gate] = {}
        for idx, token in enumerate(tokens, 1):
            gateLib[gate][header[idx - 1]] = token
    return gateLib


def read_edge_direction(inputfile):
    lines = [open(inputfile, 'r').read().strip('\n')][0].split('\n')
    edge_direct = []
    for line in lines:
        words = line.split(' ')
        edge_direct.append([words[0], words[1]])
    return edge_direct


def load_sensor_info(inputfile):
    lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
    sensorLib = {}
    for line in lines[1:]:
        tokens = line.split('\t')
        sensor = tokens[0]
        REU_off = float(tokens[1])
        REU_on = float(tokens[2])
        sensorLib[sensor] = {'on': REU_on, 'off': REU_off}
    return sensorLib


def sigmoid(ymin, ymax, Kd, n, x):
    return ymin + (ymax - ymin) / (1 + math.pow((x / Kd), n))


def sigmoid_curve(xlist, ymin, ymax, Kd, n):
    ylist = [sigmoid(ymin, ymax, Kd, n, x) for x in xlist]
    return ylist


def load_logic(inputfile):
    lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
    logic = {'PTac': [], 'PTet': [], 'PBAD': []}
    for line in lines[1:]:
        tokens = line.split('\t')[1].split(',')
        logic['PTac'].append(int(tokens[0]))
        logic['PTet'].append(int(tokens[1]))
        logic['PBAD'].append(int(tokens[2]))
    return logic


def partition_circuit(circuit_name: str, nonprimitives: List[str]) -> tuple:
    '''

    Args:
        circuit_name:
        nonprimitives:

    Returns:

    '''
    input_fp = f'./COI/{circuit_name}.edgelist'
    graph = nx.read_edgelist(input_fp, nodetype=str)
    edge_direction = read_edge_direction(input_fp)
    partition_output = find_best_cut(graph, nonprimitives)
    return graph, edge_direction, partition_output


def calc_circuit_score(assignedLib, gateLib, sensorLib):
    logic = load_logic('data/statelogic.txt')

    # initialize a score dictionary
    scoreDict = {}
    for node in assignedLib:
        if assignedLib[node]['type'] != 'input':
            scoreDict[node] = {'logic': [], 'output REU': []}

    for i in range(8):
        # print('state ', i)
        visited = 0
        assignedLib_i = copy.deepcopy(assignedLib)  # set up a temporary library

        # first calculate the input REU
        for node in assignedLib_i:
            if assignedLib_i[node]['type'] == 'input':
                if logic[assignedLib_i[node]['sensor']][i] == 0:
                    assignedLib_i[node]['output REU'] = assignedLib_i[node]['REU OFF']
                    assignedLib_i[node]['logic'] = 0
                else:
                    assignedLib_i[node]['output REU'] = assignedLib_i[node]['REU ON']
                    assignedLib_i[node]['logic'] = 1
                # print(i, node, assignedLib_i[node]['sensor'], assignedLib_i[node])
                visited += 1

        # calculate the REU of primitive and output node
        for node in assignedLib_i:
            if len(assignedLib_i[node]['in']) == 1:
                assignedLib_i[node]['visited'] = [0]
                assignedLib_i[node]['logic'] = [-1]
            elif len(assignedLib_i[node]['in']) == 2:
                assignedLib_i[node]['visited'] = [0, 0]
                assignedLib_i[node]['logic'] = [-1, -1]
        r = 1
        while visited != len(assignedLib_i.keys()):
            # print('round ###########################################', r)
            for node in assignedLib_i:
                if assignedLib_i[node]['output REU'] == 0:
                    # print('node', node)
                    # get input REU
                    # print('incoming nodes', assignedLib_i[node]['in'])
                    in_nodes = assignedLib_i[node]['in']
                    for idx, in_node in enumerate(in_nodes):
                        # print('input node', in_node, assignedLib_i[in_node]['output REU'], assignedLib_i[in_node]['logic'])
                        # if the in node has a calculated output REU
                        if assignedLib_i[in_node]['output REU'] != 0 and assignedLib_i[node]['visited'][idx] != 1:
                            # print(in_node, assignedLib_i[in_node]['output REU'])
                            assignedLib_i[node]['input REU'].append(assignedLib_i[in_node]['output REU'])
                            assignedLib_i[node]['visited'][idx] = 1
                            assignedLib_i[node]['logic'][idx] = assignedLib_i[in_node]['logic']

                    # print('inputREU', assignedLib_i[node]['input REU'])
                    # print(assignedLib_i[node]['visited'])
                    # output REU
                    if 0 not in assignedLib_i[node]['visited']:
                        if assignedLib[node]['type'] != 'output':
                            params = assignedLib_i[node]['params']
                            x = sum(assignedLib_i[node]['input REU'])
                            # print('x', x)
                            # print('params', params)
                            assignedLib_i[node]['output REU'] = sigmoid(params[0], params[1], params[2], params[3], x)
                            if 1 in assignedLib_i[node]['logic']:
                                assignedLib_i[node]['logic'] = 0
                            else:
                                assignedLib_i[node]['logic'] = 1
                        # print('output REU', assignedLib_i[node]['output REU'])
                        # print('logic of gate', node, assignedLib_i[node]['logic'])
                        # print('number of gates that have output REU', visited)
                        else:
                            assignedLib_i[node]['output REU'] = sum(assignedLib_i[node]['input REU'])
                            if 1 in assignedLib_i[node]['logic']:
                                assignedLib_i[node]['logic'] = 1
                            else:
                                assignedLib_i[node]['logic'] = 0
                        # print('done')
                        # update score dictionary

                        scoreDict[node]['logic'].append(assignedLib_i[node]['logic'])
                        scoreDict[node]['output REU'].append(assignedLib_i[node]['output REU'])
                        visited += 1
            r += 1

    # calculate score of this permutation
    Smin = 1e3
    for node in assignedLib:
        if assignedLib[node]['type'] == 'output':
            # print(node)
            # print(scoreDict[node]['output REU'])
            # print(scoreDict[node]['logic'])
            maxOFF = max([scoreDict[node]['output REU'][s] for s in range(8) if scoreDict[node]['logic'][s] == 0])
            minON = min([scoreDict[node]['output REU'][s] for s in range(8) if scoreDict[node]['logic'][s] == 1])
            # print('min on', minON, 'max off', maxOFF)
            S = minON / maxOFF
            if S < Smin:
                Smin = S
    # return the lowest S
    return Smin


@jit(nopython=False, forceobj=True)
def assign_gates(input_graph, edge_direction, partition, nonprimitives):
    # assign biological gates to partitioned graphs
    gate_library = load_gate_info('data/gatefunction.txt')
    sensor_library = load_sensor_info('data/sensor.txt')
    assigned_library = {}
    # add 'input node' and 'output node' of each gate
    for v in input_graph.nodes():
        assigned_library[v] = {}
        assigned_library[v]['in'], assigned_library[v]['out'] = [], []
        assigned_library[v]['input REU'] = []
        assigned_library[v]['output REU'] = 0

    for e in edge_direction:
        assigned_library[e[0]]['out'].append(e[1])
        assigned_library[e[1]]['in'].append(e[0])

    # assign input nodes
    input_nodes = ['x', 'y', 'cin']
    output_nodes = ['A', 'cout']
    sensors = ['PTac', 'PTet', 'PBAD']
    for idx, v in enumerate(input_nodes):
        assigned_library[v]['REU ON'] = sensor_library[sensors[idx]]['on']
        assigned_library[v]['REU OFF'] = sensor_library[sensors[idx]]['off']
        assigned_library[v]['sensor'] = sensors[idx]
        assigned_library[v]['type'] = 'input'
    for v in output_nodes:
        assigned_library[v]['type'] = 'output'

    # copy another G, remove input and output nodes
    graph_copy = copy.deepcopy(input_graph)
    for node in nonprimitives:
        graph_copy.remove_node(node)

    ## initialize the gate assignment
    for i in range(0, max(partition[1]) + 1):
        # print(partition[1])
        node_index = [a for a, b in enumerate(partition[1]) if b == i]
        graph_nodes = [list(graph_copy.nodes())[n] for n in node_index]
        # update partition result
        for v in graph_nodes:
            assigned_library[v]['part'] = i  # partition of this gate
        # assign repressor gate
        chosen_gates = random.sample(range(1, 13), len(graph_nodes))
        for idx, v in enumerate(graph_nodes):
            vg = chosen_gates[idx]
            vg_rbs = [key for key in gate_library if
                      key.split('-')[0] == str(vg)]  # different rbs choices for this repressor gate
            vg = random.choice(vg_rbs)
            assigned_library[v]['gate'] = vg
            assigned_library[v]['params'] = [
                float(gate_library[vg]['ymin']),
                float(gate_library[vg]['ymax']),
                float(gate_library[vg]['K']),
                float(gate_library[vg]['n']),
            ]
            assigned_library[v]['type'] = 'primitive'

    # calculate circuit score of initial assignments
    S0 = calc_circuit_score(assigned_library, gate_library, sensor_library)

    # simulated annealing
    Tmax = 100  # starting temperature
    C = 1e-5

    for i in range(3):  # for 100 trajectories
        SList = []
        Smax = S0
        bestLib = copy.deepcopy(assigned_library)

        for index, t in enumerate(range(3000)):  # perform simulated annealing
            print(index)
            # clone a assignedLib
            assignedLib_tmp = copy.deepcopy(assigned_library)
            # print('new tmp assignedLib', assignedLib_tmp)
            # randomly select a node
            g_swap = random.choice(list(graph_copy.nodes()))
            g_part = assignedLib_tmp[g_swap]['part']
            # print(g_swap, g_part, assignedLib_tmp[g_swap]['gate'], 'gate to be swapped')
            # get all available gates (a gate already in this partition, a different gate with a never used repressor from the library)
            # print('gates within the gate library', list(gateLib.keys()))
            used_gates = [assignedLib_tmp[g]['gate'] for g in assignedLib_tmp if
                          assignedLib_tmp[g]['type'] == 'primitive' and assignedLib_tmp[g][
                              'part'] == g_part and g != g_swap]
            # print('other used gate within this partition', used_gates)
            used_repr = [r.split('-')[0] for r in used_gates]
            # print('other used repressor within this partition', used_repr)
            # remove repressors that have already been used by other nodes in this partition
            availgates = list(set(gate_library.keys()) - set([g for g in gate_library if g.split('-')[0] in used_repr]))
            availgates.remove(assignedLib_tmp[g_swap]['gate'])
            # add other gates that have been used in this partition
            availgates = availgates + used_gates
            # print('available gates', availgates)

            # swap two gates g_swap and g_swap_f
            g_swap_t = random.choice(availgates)
            # print('gate swapped to', g_swap_t)
            if g_swap_t in used_gates:  # if swapping two gates within the same partition
                # update the gate and transfer function params of the swapped gate
                g_swap_f = [g for g in assignedLib_tmp if
                            assignedLib_tmp[g]['type'] == 'primitive' and assignedLib_tmp[g]['part'] == g_part and
                            assignedLib_tmp[g]['gate'] == g_swap_t][0]
                assignedLib_tmp[g_swap_f]['gate'] = assignedLib_tmp[g_swap]['gate']
                assignedLib_tmp[g_swap_f]['params'] = [float(gate_library[assignedLib_tmp[g_swap]['gate']]['ymin']),
                                                       float(gate_library[assignedLib_tmp[g_swap]['gate']]['ymax']),
                                                       float(gate_library[assignedLib_tmp[g_swap]['gate']]['K']),
                                                       float(gate_library[assignedLib_tmp[g_swap]['gate']]['n'])]
                # update the transfer function params of the chosen gate
                assignedLib_tmp[g_swap]['gate'] = g_swap_t
                assignedLib_tmp[g_swap]['params'] = [float(gate_library[g_swap_t]['ymin']),
                                                     float(gate_library[g_swap_t]['ymax']),
                                                     float(gate_library[g_swap_t]['K']),
                                                     float(gate_library[g_swap_t]['n'])]
            # print(assignedLib[g_swap])
            # print(assignedLib_tmp[g_swap])
            # print(assignedLib[g_swap_f])
            # print(assignedLib_tmp[g_swap_f])
            else:  # if swapping with a gate in the gate library
                assignedLib_tmp[g_swap]['gate'] = g_swap_t
                # update the transfer function params of the chosen gate
                assignedLib_tmp[g_swap]['params'] = [float(gate_library[g_swap_t]['ymin']),
                                                     float(gate_library[g_swap_t]['ymax']),
                                                     float(gate_library[g_swap_t]['K']),
                                                     float(gate_library[g_swap_t]['n'])]
            # print(assignedLib[g_swap])
            # print(assignedLib_tmp[g_swap])

            # calculate the S score, store it
            S = calc_circuit_score(assignedLib_tmp, gate_library, sensor_library)
            # print('score', S, 'original score', S0)

            # choose to accept or reject the choice

            Ti = Tmax * (math.exp(-C * t))
            if Ti > 0:
                try:
                    P = math.exp((S - S0) / Ti)
                except OverflowError:
                    P = 2
            # print('temperature', Ti)
            # print('Probability', P)
            else:
                P = math.exp((S - S0) / 0.01)

            # append the highest score that occurs to this point
            if S > Smax:
                Smax = S
                bestLib = copy.deepcopy(assignedLib_tmp)
            SList.append(Smax)
            # print('highest score that occurs to round', t, SList)

            # if P > 1, accept change
            if P > 1:
                assigned_library = assignedLib_tmp
            # print('P>1, accepted')
            # if P < 1, accept change based on probability
            else:
                if random.random() < P:
                    assigned_library = assignedLib_tmp
                # print('P<1, accepted')
            # else:
            # print('P<1, rejected')

        # print('assigned library after round', t, assignedLib)

        # record the max S
        try:
            outfile = open('./COI/adder/S_trajectory.txt', 'a')
        except FileNotFoundError:
            outfile = open('./COI/adder/S_trajectory.txt', 'w')
        outfile.write(str(i) + '\t')
        outfile.write(','.join([str(S) for S in SList]) + '\n')
        # record the best library
        record_library(bestLib, './COI/adder/optimal gate assignments_trajectory_' + str(i) + '.txt')

    return input_graph


def record_library(assignedLib, outfile):
    """ record the library with gate assignments that generate the highest circuit score """
    f_out = open(outfile, 'w')
    f_out.write('node\ttype\tsensor/gate\tparams\tpartition\n')
    for node in assignedLib:
        if assignedLib[node]['type'] == 'input':
            f_out.write('\t'.join([node, assignedLib[node]['type'], assignedLib[node]['sensor'],
                                   str([assignedLib[node]['REU ON'], assignedLib[node]['REU OFF']]), 'na']) + '\n')
        elif assignedLib[node]['type'] == 'primitive':
            f_out.write('\t'.join(
                [node, assignedLib[node]['type'], assignedLib[node]['gate'], str(assignedLib[node]['params']),
                 str(assignedLib[node]['part'])]) + '\n')
        else:
            f_out.write('\t'.join([node, assignedLib[node]['type'], 'na', 'na', 'na']) + '\n')


def plot_circuit_score(inputfile):
    """ plot circuit score trajectories in simulated annealing """
    lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    for line in lines:
        SList = [float(s) for s in line.split('\t')[1].split(',')]
        ax.plot(range(len(SList)), SList, linewidth=0.2)
    plt.show()


if __name__ == '__main__':
    inputs_and_outputs = ['cin', 'x', 'y', 'A', 'cout']  # the input and output nodes
    G, edge_direct, part_opt = partition_circuit('adder', inputs_and_outputs)

    assign_gates(G, edge_direct, part_opt, inputs_and_outputs)
    plot_circuit_score('./COI/adder/C_1e-5_t_30k/S_trajectory.txt')
