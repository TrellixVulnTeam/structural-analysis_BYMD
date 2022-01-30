import matplotlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# number output formatting
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=2000)
np.set_printoptions(precision=0)
nodes = [] # listof Nodes i.e., [[x1, y1], [x2, y2]]
members = [] # listof Member lengths
loads = []


class Member:
    '''
    Fields:
        Length (Nat),
        Theta (Nat),
        Global DOF (listof Nat),
        Local DOF (listof Nat),
        DOF Restraints (listof Str)
    '''

    def __init__(self, number1):
        '''
        Constructor: Create a Member object by
        calling Member(start_x, end_x, start_y, end_y, theta, global_dof, local_dof, restraints)

        Effects: Mutates Self

        __init__: Member Nat Nat Nat Nat Nat (listof Nat) (listof Nat) (listof Str) -> None
        '''

        self.label = st.markdown("Member " + str(number1))
        self.name = "Member #" + str(number1)

        self.col1, self.col2, self.col3, self.col4, self.col5, self.col6, self.col7, self.col8 = st.columns(8)
        with self.col1:
            self.start_node = float(st.text_input('Start', value=0, key="node_start_"+str(number1)))

        with self.col2:
            self.end_node = float(st.text_input('End', value=0, key="node_end_" + str(number1)))

        with self.col3:
            self.x_start_dof = float(st.text_input("x DOF start", value=0, key="dof_start_x" + str(number1)))

        with self.col4:
            self.y_start_dof = float(st.text_input("y DOF  start", value=0, key="dof_start_y"+ str(number1)))

        with self.col5:
            self.z_start_dof = float(st.text_input("z DOF start", value=0, key="dof_start_z"+ str(number1)))

        with self.col6:
            self.x_end_dof = float(st.text_input("x DOF end", value=0, key="dof_end_x"+ str(number1)))

        with self.col7:
            self.y_end_dof = float(st.text_input("y DOF end", value=0, key="dof_end_y"+ str(number1)))

        with self.col8:
            self.z_end_dof = float(st.text_input("z DOF end ", value=0, key="dof_end_z"+ str(number1)))

        self.start_x = nodes[int(self.start_node) - 1].x
        self.start_y = nodes[int(self.start_node) - 1].y
        self.end_x = nodes[int(self.end_node) - 1].x
        self.end_y = nodes[int(self.end_node) - 1].y
        self.id = number1
        self.member_length = math.sqrt((self.end_x-self.start_x) ** 2 + (self.end_y-self.start_y) ** 2)
        self.start_node_dof = [self.x_start_dof, self.y_start_dof, self.z_start_dof]
        self.end_node_dof = [self.x_end_dof, self.y_end_dof, self.z_end_dof]

        if self.start_x == self.end_x:
                if self.start_y < self.end_y:
                    self.theta = 90  #* np.pi / 180
                else:
                    self.theta = -90
        else:
            self.theta = math.atan((self.end_y-self.start_y)/(self.end_x-self.start_x)) * 180 / np.pi
        self.label = st.sidebar.markdown('m' + str(number1) + ":" + f"  L = {str(self.member_length)}m," + f" Angle = {str(self.theta)}" )

    def __repr__(self):
        s = "Member #{0.id}"
        return s.format(self)

    def member_matrix(self):
        AE = 548000    # AE
        EI = 16120000  # EI
        #self.member_length = 5
        if self.start_x == self.end_x:
                if self.start_y < self.end_y:
                    self.theta = 90 #* np.pi / 180
                else:
                    self.theta = -90
        else:
            self.theta = math.atan((self.end_y-self.start_y)/(self.end_x-self.start_x)) * 180 / np.pi
           # print(self.theta)
        #row1 = [(AE/self.member_length) * (np.cos(self.theta)) ** 2 + (12 * EI / self.member_length ** 3) * (np.sin(self.theta)) ** 2]
        member_stiffness_matrix = np.array([
            [((AE / self.member_length) * (np.cos(self.theta * np.pi / 180)) ** 2) + (
                    (12 * EI) / (self.member_length ** 3)) * (np.sin(self.theta * np.pi / 180)) ** 2,
             ((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                     np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)),
             (-6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180), -(
                    ((AE / self.member_length) * (np.cos(self.theta * np.pi / 180)) ** 2) + (
                    (12 * EI) / (self.member_length ** 3)) * (np.sin(self.theta * np.pi / 180)) ** 2),
             -((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                     np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)),
             (-6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180)],
            [((AE /self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                    np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)),
             ((AE / self.member_length) * (np.sin(self.theta * np.pi / 180)) ** 2) + (12 * EI / (self.member_length ** 3)) *
             (np.cos(self.theta * np.pi / 180)) ** 2,
             (6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180), -(
                    ((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                    np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180))), -1 * (
                     ((AE / self.member_length) * (np.sin(self.theta * np.pi / 180)) ** 2) + (
                     (12 * EI) / (self.member_length ** 3)) * (np.cos(self.theta * np.pi / 180)) ** 2),
             (6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180)],
            [(-6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180),
             (6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180), (4 * EI / self.member_length),
             (6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180),
             (-6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180), (2 * EI / self.member_length)],
            [-(((AE / self.member_length) * (np.cos(self.theta * np.pi / 180)) ** 2) + (
                    (12 * EI) / (self.member_length ** 3)) * (np.sin(self.theta * np.pi / 180)) ** 2), -(
                    ((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                    np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180))),
             (6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180), (
                     ((AE / self.member_length) * (np.cos(self.theta * np.pi / 180)) ** 2) + (
                     (12 * EI) / (self.member_length ** 3)) * (np.sin(self.theta * np.pi / 180)) ** 2),
             ((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                     np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)),
             (6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180)],
            [-((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                    np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)), -1 * (
                     ((AE / self.member_length) * (np.sin(self.theta * np.pi / 180)) ** 2) + (
                     (12 * EI) / (self.member_length ** 3)) * (np.cos(self.theta * np.pi / 180)) ** 2),
             (-6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180),
             ((AE / self.member_length) - (12 * EI / (self.member_length ** 3))) * (
                     np.cos(self.theta * np.pi / 180) * np.sin(self.theta * np.pi / 180)),
             ((AE / self.member_length) * (np.sin(self.theta * np.pi / 180)) ** 2) + (12 * EI / (self.member_length ** 3)) * (
                 np.cos(self.theta * np.pi / 180)) ** 2,
             (-6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180)],
            [(-6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180),
             (6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180), 2 * EI / self.member_length,
             (6 * EI / (self.member_length ** 2)) * np.sin(self.theta * np.pi / 180),
             (-6 * EI / (self.member_length ** 2)) * np.cos(self.theta * np.pi / 180), (4 * EI / self.member_length)]])

        return member_stiffness_matrix


class Node:
    def __init__(self, number):

        self.name = "n" + str(number)
        st.write('Node ID #' + str(number))
        self.col1, self.col2, self.col3 = st.columns(3)
        with self.col1:
            self.x = float(st.text_input('X', value=0, key=str(number)))
        with self.col2:
            self.y = float(st.text_input('Y', value=0, key=str(number+(number/0.23))))
        with self.col3:
            self.has_support = st.selectbox('Support?', ('True', 'False'), key = str(number+(number/0.25123)))

        self.label = st.sidebar.markdown('Node ' + str(number) + ":" + f" ({self.x}, {self.y})")

    def __repr__(self):
        s = "NODE #" + str(number) + f" ({self.x},{self.y})"
        return s


class Load:
    '''
        Fields:
            Magnitude (Nat),
            Orientation [vertical/horizontal] (Nat),
            Member ID (Nat),
            Location (Nat)
        '''
    def __init__(self, number2):
        self.name = "Load" + str(number2)
        self.col1, self.col2 = st.columns(2)
        with self.col1:
            self.magnitude = float(st.text_input('Load [kN]', value=0, key=str(number2)))
        with self.col2:
            self.dof = float(st.text_input('Applied at DOF #:', value =1, key = str(number2)))

        self.label = st.sidebar.markdown("Load applied at DOF #" + str(self.dof))

    def __repr__(self):
        s = "{0.orientation} load applied {0.location}m along member {0.member_id}"
        return s.format(self)


def get_end_dof(lom):
    end_dof = []
    for i in range(len(lom)):
        end_dof.append(lom[i].end_node_dof)
    return end_dof


def get_start_dof(lom):
    start_dof = []
    for i in range(len(lom)):
        start_dof.append(lom[i].start_node_dof)
    return start_dof


def global_stiffness_matrix_fwd_pass(lom):
    # consumes list of members
    # checks if beginning/ending dof's are the same
    gsm = np.zeros(((len(lom)+1)*3, (len(lom)+1)*3))  # need to fix, global stiffness matrix changes
    end_dof = get_end_dof(lom)
    start_dof = get_start_dof(lom)
    for i in range(len(lom)):
        for j in range(len(lom)):
            print("Comparing DOF")
            print(lom[i].start_node_dof, lom[j].end_node_dof)
            if lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) > 1 and j == end_dof.index(lom[i].start_node_dof):
                start = int(lom[i].start_node_dof[0])
                stop = int(lom[i].start_node_dof[-1])
                gsm[start-1:stop, start-1:stop] += lom[i].member_matrix()[0:3, 0:3] + lom[j].member_matrix()[3:,3:]
                print("Condition 1 ran")
            elif lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) > 1:
                start = int(lom[i].start_node_dof[0])
                stop = int(lom[i].start_node_dof[-1])
                gsm[start-1:stop, start-1:stop] +=  lom[j].member_matrix()[3:,3:]
                print("Condition 2 ran")
            elif lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) == 1:
                start = int(lom[i].start_node_dof[0])
                stop = int(lom[i].start_node_dof[-1])
                print("lom[i]")
                print(lom[i].member_matrix())
                print("lom[j]")
                print(lom[j].member_matrix())
                gsm[start - 1:stop, start - 1:stop] += np.add(lom[i].member_matrix()[0:3, 0:3], lom[j].member_matrix()[3:, 3:])
                print("Condition 3 ran")
            elif lom[i].start_node_dof != lom[j].end_node_dof and j == len(lom) - 1 and lom[i].start_node_dof not in end_dof:
                start = int(lom[i].start_node_dof[0])
                stop = int(lom[i].start_node_dof[-1])
                gsm[start - 1:stop, start - 1:stop] += lom[i].member_matrix()[0:3, 0:3]
                print("Condition 4 ran")
            print(gsm)
            print("----------------------------")
    start = int(max(end_dof)[0] - 1)
    stop = int(max(end_dof)[-1])
    #print(start, stop)
    gsm[start:stop, start:stop] += lom[end_dof.index(max(end_dof))].member_matrix()[3:, 3:] # have to fix, it's hardcoded
    return gsm


def global_stiffness_matrix_bwd_pass(lom):
    start_dof = get_start_dof(lom)
    end_dof = get_end_dof(lom)
    gsm = global_stiffness_matrix_fwd_pass(lom)

    for i in range(len(lom)):
        start_col = int(lom[i].start_node_dof[0])
        stop_col =  int(lom[i].start_node_dof[-1])
        start_row = int(lom[i].end_node_dof[0])
        stop_row =  int(lom[i].end_node_dof[-1])
        gsm[start_row-1: stop_row, start_col-1:stop_col] += lom[i].member_matrix()[3:, 0:3]
        #gsm[start_col - 1:stop_col, start_row-1: stop_row] += lom[i].member_matrix()[3:, 0:3]

    for i in range(len(lom)):
        start_col = int(lom[i].end_node_dof[0])
        stop_col =  int(lom[i].end_node_dof[-1])
        start_row = int(lom[i].start_node_dof[0])
        stop_row =  int(lom[i].start_node_dof[-1])
        gsm[start_row-1: stop_row, start_col-1:stop_col] += lom[i].member_matrix()[0:3, 3:]
    start = int(max(end_dof)[0] - 1)
    stop = int(max(end_dof)[-1])
    print(start, stop)
    #gsm[start:stop, start:stop] += lom[end_dof.index(max(end_dof))].member_matrix()[3:,3:]  # have to fix, it's hardcoded

    return gsm

def show_properties(lst):
    for i in range(len(lst)):
        print(f"Member {lst[i].member_id}, L = {lst[i].member_length}, theta = {lst[i].theta}")
        print("Member Stiffness Matrix")
        print(lst[i].member_matrix())
        print("_______________________________________________")
        print("                                               ")


def assemble_displacement_matrix(lom):
    count_joints = 0
    for i in range(len(lom)):
        if not lom[i].has_support:
            count_joints += 1
    return (count_joints+1)*3


def add_node(number):
    for i in range(1, number+1):
        nodes.append(Node(i))


def add_member(number1):
    for j in range(1, number1+1):
        members.append(Member(j))


def add_load(number2):
    for k in range(1, number2+1):
        loads.append(Load(k))


def compute_displacements(lon, loads, gsm):
    num_joints = 0
    for node in lon:
        if node.has_support == "False":
            num_joints += 1
        print(node.has_support)
    Qk = np.zeros((num_joints*3)) #Dk = np.array((num_joints*3, 1)) not necessary for now since Dk will always just be zeros if there is no settlement etc.
    for load in loads:
        Qk[int(load.dof) - 1] -= -int(load.magnitude)
    KFF = gsm[0:num_joints*3, 0:num_joints*3]
    KFF_inv = np.linalg.inv(KFF)
    displacements = np.matmul(KFF_inv, Qk) #* 1000
    print(["{:0.5f}".format(x) for x in displacements])
    return displacements


def compute_forces(lon, loads, gsm):
    num_joints = 0
    for node in lon:
        if node.has_support == "False":
            num_joints += 1
        print(node.has_support)

    KEF = gsm[num_joints*3:, 0:num_joints*3]
    Du = compute_displacements(lon, loads, gsm)
    print("KEF", KEF)
    print("Du", Du)
    print("Pigglz", num_joints)
    return np.matmul(KEF, Du)


def show_structure(lon, lom):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_aspect(1)
    points = []
    x_val = []
    y_val = []

    for i in range(len(lom)):
        ax.plot([lom[i].start_x, lom[i].end_x], [lom[i].start_y, lom[i].end_y], 'k-')

    #for i in range(1, len(points)):
    #    #ax.plot([points[i-1][0], points[i][0]], [points[i-1][1], points[i][1]], 'k-')
    #    ax.plot([x_val[i], ], [], 'k-')
    #    ax.set(xlim=(min(x_val) - 5, 5 + max(x_val)), ylim=(min(y_val) - 5, 5 + max(y_val)))
    #    print("P1: ", points[i-1], "P2: ", points[i])
#
    for j in range(len(lon)):
        if lon[j].has_support == "True":
            ax.plot([lon[j].x - 0.5, lon[j].x + 0.5], [lon[j].y, lon[j].y], color = 'k')


    st.pyplot(fig)
    plt.show()


def show_loads(lol):

    pass


st.title("Structural 2D Analysis")
st.sidebar.title("System Information")
col1, col2 = st.columns(2)

with col1:
    number = math.floor(st.number_input('Click "+" to add nodes', min_value=0, value=0, step=1))

with col2:
    st.empty()

add_node(number)

col3, col4 = st.columns(2)

with col3:
    number1 = math.floor(st.number_input('Click "+" to add Members', min_value=0, value=0, step=1, key="member"))

with col4:
    st.empty()

add_member(number1)
print(nodes)
print(members)

show = st.checkbox("Show structure")

if show:
    show_structure(nodes,members)

answer_stiffness_matrices = st.button("Compute stiffness matrices")

if answer_stiffness_matrices and len(members) >= 1:
    for i in range(len(members)):
        df = pd.DataFrame(members[i].member_matrix())
        st.write("Member #" + str(i + 1) + " - Stiffness Matrix")
        st.table(df.round(1))
    df = global_stiffness_matrix_bwd_pass(members)
    st.write("Structural System Stiffness Matrix")
    st.table(df)

if len(nodes) >= 2 and len(members) >= 1:
    gsm = global_stiffness_matrix_bwd_pass(members)

col5, col6 = st.columns(2)

with col5:
    number2 = math.floor(st.number_input('Click "+" to add nodal loads', min_value=0, value=0, step=1, key="loads"))

with col4:
    st.empty()

add_load(number2)

answer_compute_forces = st.button("Solve system")

if answer_compute_forces:
    displacements = compute_displacements(nodes, loads, gsm)
    forces = compute_forces(nodes, loads, gsm)
    st.write("Displacement Matrix")
    st.write(displacements)
    st.write("Forces")
    st.write(forces)

for i in range(len(nodes)):
    print("X", nodes[i].x, "Y", nodes[i].y)
