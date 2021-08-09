import matplotlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import SessionState
import math

st.header("Frame Calculator v1.0")
st.subheader("by Ali Chaudhry, BASc. Candidate")
st.write("Please see bottom of page for limitations.")
ss = SessionState.get(n=1)
number = ss.n

# stores node information in a list (max 4 nodes for now)
nodes = []

# stores loading information in a list
loads = []
load_posn = [] # posn indicating at which node the load was applied to
load_dist = [] # how far away each load is from the node
load_mag = []  # magnitude of load
load_type = [] # type of load (i.e., moment, vertical, or lateral load)

# x-values of nodes
x_val = []

# y-values of nodes
y_val = []

# member lengths
member_length = []

# member inclinations
theta = []

st.sidebar.title("Nodal Information")
ax = plt.axes()

colEI, colAE = st.columns(2)

with colEI:
    EI = float(st.text_input("Enter EI: ", value=105000, key="EI"))
with colAE:
    AE = float(st.text_input("Enter AE: ", value=6300000, key="AE"))


class Node:
    def __init__(self, number):
        self.label = st.sidebar.markdown('Node ' + str(number))
        self.name = "n" + str(number)
        self.col1, self.col2 = st.sidebar.columns(2)
        with self.col1:
            self.x = float(st.text_input('X', value=0, key=str(number)))
        with self.col2:
            self.y = float(st.text_input('Y', value=0, key=str(number+(number/0.23))))


class Load:
    def __init__(self, num):
        self.label = st.markdown('Load ' + str(num))
        self.name = "n" + str(num)
        self.col1, self.col2, self.col3, self.col4 = st.columns(4)
        with self.col1:
            self.magnitude = float(st.text_input('Load [kN]', value=0, key="load"+str(num)))
        with self.col2:
            self.position = float(st.text_input('@ Node #', value=0, key="posn"+str(num)))
        with self.col3:
            self.dist = float(st.text_input('Distance on member:',value=0, key="dist"+str(num)))
        with self.col4:
            self.load_type = st.selectbox("Load Type", ('Vertical', 'Lateral', 'Moment'), key="type"+str(num))

def add_node(number):
    for i in range(1, number+1):
        nodes.append(Node(i))


def add_load(number):
    for i in range(0,number+1):
        loads.append(Load(i))


def posn_extract(number):
    for j in range(0, number+1):
        load_posn.append(loads[j].position)


def dist_extract(number):
    for i in range(0, number+1):
        load_dist.append(loads[i].dist)


def magnitude_extract(number):
    for k in range(0, number+1):
        load_mag.append(loads[k].magnitude)


def load_type_extract(number):
    for i in range(0, number+1):
        load_type.append(loads[i].load_type)


def load_plot(load_posn):
    for i in range(0, len(load_posn)):
        if load_type[i] == 'Vertical':
            plt.arrow(nodes[math.floor(load_posn[i])].x + load_dist[i],
                      nodes[math.floor(load_posn[i])].y * 1.5,
                      0,
                      -(nodes[math.floor(load_posn[i])].y * 1.5 - nodes[math.floor(load_posn[i])].y),
                      length_includes_head=True,
                      color="red", head_width=0.25, head_length=0.35)
        elif load_type[i] == 'Lateral':
            plt.arrow(nodes[math.floor(load_posn[i])].x,
                      nodes[math.floor(load_posn[i])].y,
                      1.4 ,
                      0,
                      length_includes_head = True,
                      color="red", head_width = 0.25, head_length = 0.35)




coli1, coli2 = st.sidebar.columns(2)
colj1, colj2 = st.columns(2)

with coli1:
    number = math.floor(st.number_input("Click '+' to add nodes", step=1, value=0))
with coli2:
    st.empty()

with colj1:
    num_load = math.floor(st.number_input("Click '+' to add loads", step=1, value=0))
with colj2:
    st.empty()

add_node(number)
fig = plt.figure()

if len(nodes) > 0:
    add_load(num_load)
    posn_extract(num_load)
    dist_extract(num_load)
    magnitude_extract(num_load)
    load_type_extract(num_load)
    for i in range(0, len(nodes)):
        x_val.append(nodes[i].x)
        y_val.append(nodes[i].y)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    plt.plot(x_val, y_val, color='k')
    ax.set(xlim=(min(x_val) - 5, 5 + max(x_val)), ylim=(min(y_val) - 5, 5 + max(y_val)))

    load_plot(load_posn)

fixed = st.checkbox("Fixed")

if fixed:
    plt.plot([nodes[0].x - 0.5, nodes[0].x + 0.5], [0, 0], color='k')
    plt.plot([nodes[-1].x - 0.5, nodes[-1].x + 0.5], [nodes[-1].y, nodes[-1].y], color='k')

if len(nodes) >= 2:
    for i in range(1, len(nodes)):
        dx = nodes[i].x - nodes[i-1].x
        dy = nodes[i].y - nodes[i-1].y
        if dy == 0 and dx != 0:
            theta.append(0)
            member_length.append(abs(dx))
        elif dx == 0 and (nodes[i].x == 0 and nodes[i-1].y == 0):
            theta.append(90)
            member_length.append(abs(dy))
        elif dy != 0 and dx != 0:
            theta.append((np.arctan(dy / dx)) * (180 / np.pi))
            member_length.append((dx ** 2 + dy ** 2) ** 0.5)
        else:
            theta.append(-90)
            member_length.append(abs(dy))

member_matrices = {}
for i in range(len(member_length)):
    member_matrices[f"m{i+1}"] = []

local_q = {}

for i in range(len(member_length)):
    if member_length[i] > 0:
        member_matrices[f'm{i + 1}'] = np.array([
            [((AE / member_length[i]) * (np.cos(theta[i] * np.pi / 180)) ** 2) + (
                    (12 * EI) / (member_length[i] ** 3)) * (np.sin(theta[i] * np.pi / 180)) ** 2,
             ((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                     np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)),
             (-6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180), -(
                    ((AE / member_length[i]) * (np.cos(theta[i] * np.pi / 180)) ** 2) + (
                    (12 * EI) / (member_length[i] ** 3)) * (np.sin(theta[i] * np.pi / 180)) ** 2),
             -((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                     np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)),
             (-6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180)],
            [((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                    np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)),
             ((AE / member_length[i]) * (np.sin(theta[i] * np.pi / 180)) ** 2) + (12 * EI / (member_length[i] ** 3)) *
             (np.cos(theta[i] * np.pi / 180)) ** 2,
             (6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180), -(
                    ((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                    np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180))), -1 * (
                     ((AE / member_length[i]) * (np.sin(theta[i] * np.pi / 180)) ** 2) + (
                     (12 * EI) / (member_length[i] ** 3)) * (np.cos(theta[i] * np.pi / 180)) ** 2),
             (6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180)],
            [(-6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180),
             (6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180), (4 * EI / member_length[i]),
             (6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180),
             (-6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180), (2 * EI / member_length[i])],
            [-(((AE / member_length[i]) * (np.cos(theta[i] * np.pi / 180)) ** 2) + (
                    (12 * EI) / (member_length[i] ** 3)) * (np.sin(theta[i] * np.pi / 180)) ** 2), -(
                    ((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                    np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180))),
             (6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180), (
                     ((AE / member_length[i]) * (np.cos(theta[i] * np.pi / 180)) ** 2) + (
                     (12 * EI) / (member_length[i] ** 3)) * (np.sin(theta[i] * np.pi / 180)) ** 2),
             ((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                     np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)),
             (6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180)],
            [-((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                    np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)), -1 * (
                     ((AE / member_length[i]) * (np.sin(theta[i] * np.pi / 180)) ** 2) + (
                     (12 * EI) / (member_length[i] ** 3)) * (np.cos(theta[i] * np.pi / 180)) ** 2),
             (-6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180),
             ((AE / member_length[i]) - (12 * EI / (member_length[i] ** 3))) * (
                     np.cos(theta[i] * np.pi / 180) * np.sin(theta[i] * np.pi / 180)),
             ((AE / member_length[i]) * (np.sin(theta[i] * np.pi / 180)) ** 2) + (12 * EI / (member_length[i] ** 3)) * (
                 np.cos(theta[i] * np.pi / 180)) ** 2,
             (-6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180)],
            [(-6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180),
             (6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180), 2 * EI / member_length[i],
             (6 * EI / (member_length[i] ** 2)) * np.sin(theta[i] * np.pi / 180),
             (-6 * EI / (member_length[i] ** 2)) * np.cos(theta[i] * np.pi / 180), (4 * EI / member_length[i])]])


dof = len(nodes) * 3 # assumes support conditions are fixed-fixed, if simply supported then we need l
                     # len(nodes) * 3 - 3  (since we lose moment resistance @ pin, and moment + horizontal resistance @ roller)

vertical_dof = [1, 4]
horizontal_dof = [0, 3]
bending_dof = [2, 5]

global_matrix = np.zeros((dof, dof))
Du = np.zeros((6, 1))
Dk = np.zeros((6, 1))
QuJ = np.zeros((6, 1))

if st.sidebar.checkbox("Done"):
    for i in range(len(member_matrices)):
        if member_matrices[f"m{i + 1}"] != []:
            sub123_1 = member_matrices["m1"][3:, 3:]
            sub123_2 = member_matrices["m2"][0:3, 0:3]

            sub456_1 = member_matrices["m2"][3:, 3:]
            sub456_2 = member_matrices["m3"][0:3, 0:3]

            sub789_1 = member_matrices["m1"][0:3, 0:3]
            sub101112_1 = member_matrices["m3"][3:, 3:]

            result_123 = np.zeros((3, 3))
            result_456 = np.zeros((3, 3))

            for i in range(len(sub123_1)):
                for j in range(len(sub123_1[0])):
                    result_123[i][j] = sub123_1[i][j] + sub123_2[i][j]

            for i in range(len(sub456_1)):
                for j in range(len(sub456_1[0])):
                    result_456[i][j] = sub456_1[i][j] + sub456_2[i][j]

            global_matrix = np.zeros((dof, dof))

            global_matrix[0:3, 0:3] = result_123[:, :]
            global_matrix[3:6, 3:6] = result_456[:, :]
            global_matrix[6:9, 6:9] = sub789_1
            global_matrix[9:12, 9:12] = sub101112_1
            global_matrix[0:3, 3:6] = member_matrices["m2"][0:3, 3:6]
            global_matrix[6:9, 0:3] = member_matrices["m1"][0:3, 3:]  ####
            global_matrix[0:3, 3:6] = member_matrices["m2"][0:3, 3:6]
            global_matrix[0:3, 6:9] = member_matrices["m1"][3:, 0:3]
            global_matrix[3:6, 0:3] = member_matrices["m2"][3:, 0:3]
            global_matrix[9:, 3:6] = member_matrices["m3"][3:, 0:3]
            global_matrix[3:6, 9:] = member_matrices["m3"][0:3, 3:]  ###
            global_matrix[6:9, 6:9] = member_matrices["m1"][0:3, 0:3]
            global_matrix[9:12, 9:12] = member_matrices["m3"][3:, 3:]

            # print(global_matrix)
            x_val = []
            y_val = []

            # Displacement Vector

            if fixed:
                num_Dk = 2 * 3
                num_Du = 6
                Dk = np.array([[0], [0], [0], [0], [0], [0]])
                # change later!!!! for i in range(len(supportID)):
                #   Dk.append(0)

                KFF = global_matrix[0:num_Dk, 0:num_Dk]
                KFE = global_matrix[0:num_Dk, num_Dk:]
                KEF = np.matrix(global_matrix[num_Dk:, 0:num_Dk])
                KEE = np.matrix(global_matrix[num_Dk:, num_Dk:])

                Qj = np.array([[0], [0], [0], [0], [0], [0]])
                QFEF = np.array([[0], [0], [0], [0], [0], [0]])

                if len(loads) > 0:
                    for i in range(len(loads)):
                        for j in range(len(load_type)):
                            if load_dist[i] == 0:
                                #Qj[math.floor(load_posn[i])] = load_mag[i]
                                if load_type[i] == 'Vertical' and load_posn[i] == 1:
                                    Qj[vertical_dof[0]] = load_mag[i]
                                elif load_type[i] == 'Vertical' and load_posn[i] == 2:
                                    Qj[vertical_dof[1]] = load_mag[i]
                                elif load_type[i] == 'Lateral' and load_posn[i] == 1:
                                    Qj[horizontal_dof[0]] = load_mag[i]
                                elif load_type[i] == 'Lateral' and load_posn[i] == 2:
                                    Qj[horizontal_dof[1]] = load_mag[i]
                                elif load_type[i] == 'Moment' and load_posn[i] == 1:
                                    Qj[bending_dof[0]] = load_mag[i]
                                elif load_type[i] == 'Moment' and load_posn[i] == 2:
                                    Qj[bending_dof[1]] = load_mag[i]
                Q = Qj
                KFF_inv = np.linalg.inv(KFF)
                KFE_Dk = np.matmul(KFE, Dk)
                Qk_KFEDK = np.subtract(Q, KFE_Dk)  # Qk_KFEDK = np.subtract(Qk, KFE_Dk)
                Du = np.matmul(KFF_inv, Qk_KFEDK)
                D1 = np.matrix.transpose(np.array([[0, 0, 0, Du[0], Du[1], Du[2]]]))
                D2 = np.matrix.transpose(np.array([[Du[0], Du[1], Du[2], Du[3], Du[4], Du[5]]]))
                D3 = np.matrix.transpose(np.array([[Du[3], Du[4], Du[5], 0, 0, 0]]))

                D = {
                    "D1": D1,
                    "D2": D2,
                    "D3": D3
                }

                QuJ = np.add(np.matmul(KEF, Du), np.matmul(KEE, Dk)) + QFEF  # , QFEF) #+ QFEF

                ## Bending Moment Diagram
                for i in range(len(member_matrices)):
                    T = np.array([[np.cos(theta[i] * np.pi / 180), np.sin(theta[i] * np.pi / 180), 0, 0, 0, 0],
                                  [-np.sin(theta[i] * np.pi / 180), np.cos(theta[i] * np.pi / 180), 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, np.cos(theta[i] * np.pi / 180), np.sin(theta[i] * np.pi / 180), 0],
                                  [0, 0, 0, -np.sin(theta[i] * np.pi / 180), np.cos(theta[i] * np.pi / 180), 0],
                                  [0, 0, 0, 0, 0, 1]])  # Transformation Matrix of Member i corresponding to theta i
                    term_1 = np.matmul(member_matrices[f"m{i + 1}"], D[f"D{i + 1}"])
                    # Ignoring QFEF for now since no other loads except on nodes
                    local_q[f"m{i}"] = [np.matmul(T, term_1)]


st.pyplot(fig)

print(load_type)
colr1, colr2 = st.columns(2)

with colr1:
    st.header("Displacement Analysis:")
    for i in range(len(Du)):
        st.text(f"D" + str(i + 1) + f": {str(Du[i])}")
with colr2:
    st.header("Force Analysis:")
    for i in range(len(QuJ)):
        st.text(f"R" + str(i + 1) + f": {str(QuJ[i])}")

#Calculator is limited to:
#- 4 nodes
#- loads must be on joints
#--------------------------
#Not limited to:
#- Can calculate more than one joint load
#- User is able to put as many nodes as they wish
#- User is able to put as many loads as they wish
#- Working button
#- change frame stiffness (however, all members will have same stiffness)
#
#Will add: ")
#- Analysis for non-nodal forces")
#- BMD for angled frames")

st.subheader("Limitations: ")
st.text("This calculator is limited to simple frames with four nodes. \n"
        "The frame must also be loaded on it's joints (not compatible with \n"
        "non-nodal forces yet).  This calculator is an ongoing side-project, \n"
        "so in the future I hope to add more functionalities to generalize \n"
        "the calculations such that more frames are supported.")
