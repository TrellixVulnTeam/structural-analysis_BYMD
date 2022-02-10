##
## ***************************************************
## Muhammad Nuh Ali Reza Chaudhry
## Candidate for BASc in Structural Engineering
## Structural Analysis Webapp V1.1
## 02/05/2022
## ***************************************************
##


# import resources
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import matplotlib

matplotlib.use('Agg')


# number output formatting
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=2000)
np.set_printoptions(precision=0)

# global variables relating to the structure and user input
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
        AE = 6300000    # AE
        EI = 105000  # EI
        if self.start_x == self.end_x:
                if self.start_y < self.end_y:
                    self.theta = 90
                else:
                    self.theta = -90
        else:
            self.theta = math.atan((self.end_y-self.start_y)/(self.end_x-self.start_x)) * 180 / np.pi
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
        self.col1, self.col2, self.col3, self.col4 = st.columns(4)
        with self.col1:
            self.magnitude = float(st.text_input('Load [kN]', value=0, key=str(number2)))
        with self.col2:
            self.node = int(st.text_input('Applied at node #', value = 1, key = str(number2)))
        with self.col3:
            self.dof = float(st.text_input('Applied at DOF #:', value =1, key = str(number2)))
        with self.col4:
            self.orientation = st.selectbox("Orientation: ", ('Horizontal', 'Vertical', 'Moment'), key = str(number2))

        self.label = st.sidebar.markdown("Load applied at DOF #" + str(self.dof))



    def __repr__(self):
        s = "{0.orientation} load applied at node {0.node} at DOF #{0.dof}"
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


def get_support_dof(lom):
    support_dof = []
    for member in lom:
        if nodes[int(member.start_node) - 1].has_support == "True":
            support_dof += [member.start_node_dof]
        elif nodes[int(member.end_node) - 1].has_support == "True":
            support_dof += [member.end_node_dof]
    support_dof.pop(0)
    support_dof.pop(-1)
    return support_dof


def magic(lom, sussy_dof):
    sub_matrix = np.zeros((3,3))
    start_dof = get_start_dof(lom)
    for member in lom:
        if start_dof.count(member.start_node_dof) >= 2:
            if member.start_node_dof == sussy_dof:
                #print(member.member_matrix()[0:3, 0:3])
                sub_matrix[0:3, 0:3] += member.member_matrix()[0:3, 0:3]
    for member in lom:
        if member.end_node_dof == sussy_dof:
            sub_matrix[0:3, 0:3] += member.member_matrix()[3:, 3:]
    return sub_matrix


def get_max_dof(lom):
    max_end = 0
    max_start = 0
    start_dof = get_start_dof(lom)
    end_dof = get_end_dof(lom)
    for ending_dof in end_dof:
        for i in range(len(ending_dof)):
            if ending_dof[i] > max_end:
                max_end = ending_dof[i]
    for starting_dof in start_dof:
        for j in range(len(starting_dof)):
            if starting_dof[i] > max_start:
                max_start = starting_dof[i]
    return max(max_end, max_start)


def global_stiffness_matrix_fwd_pass(lom):
    # consumes list of members
    # checks if beginning/ending dof's are the same
    end_dof = get_end_dof(lom)
    start_dof = get_start_dof(lom)
    dof = int(get_max_dof(lom))
    gsm = np.zeros((dof, dof))

    for i in range(len(lom)):
        for j in range(len(lom)):

            #print("----------------------------------")
            #print(f"Comparing start node dof {lom[i].start_node_dof} and end node dof{lom[j].end_node_dof}")
            if lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) > 1 and j == end_dof.index(lom[i].start_node_dof):
                #print("Condition 1 ran")
                #print("lom[i] stiffness matrix")
                #print(lom[i].member_matrix())
                #print("lom[j] stiffness matrix")
                #print(lom[j].member_matrix())
                start = lom[i].start_node_dof[0]
                stop = lom[i].start_node_dof[-1]
                start = int(start)
                stop = int(stop)
                gsm[start-1:stop, start-1:stop] += lom[i].member_matrix()[0:3, 0:3] + lom[j].member_matrix()[3:,3:]
                #print("GSM after iteration")
                #print(gsm)
                #print("----------------------------------")
            elif lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) > 1:
                #print("Condition 2 ran")
                #print("lom[i] stiffness matrix")
                #print(lom[i].member_matrix())
                #print("lom[j] stiffness matrix")
                #print(lom[j].member_matrix())
                start = lom[i].start_node_dof[0]
                stop = lom[i].start_node_dof[-1]
                start = int(start)
                stop = int(stop)
                gsm[start-1:stop, start-1:stop] +=  lom[j].member_matrix()[3:,3:]
                #print("GSM after iteration")
                #print(gsm)
                #print("----------------------------------")
            elif lom[i].start_node_dof == lom[j].end_node_dof and lom[i].start_node_dof in end_dof and end_dof.count(lom[i].start_node_dof) == 1 and start_dof.count(lom[i].start_node_dof) == 1:
                #print("Condition 3 ran")
                #print("lom[i] stiffness matrix")
                #print(lom[i].member_matrix())
                #print("lom[j] stiffness matrix")
                #print(lom[j].member_matrix())
                start = lom[i].start_node_dof[0]
                stop = lom[i].start_node_dof[-1]
                start = int(start)
                stop = int(stop)
                gsm[start - 1:stop, start - 1:stop] += lom[i].member_matrix()[0:3, 0:3] + lom[j].member_matrix()[3:, 3:]
                #print("GSM after iteration")
                #print(gsm)
                #print("----------------------------------")

            elif lom[i].start_node_dof != lom[j].end_node_dof and j == len(lom) - 1 and lom[i].start_node_dof not in end_dof:
                #print("Condition 4 ran")
                #print("lom[i] stiffness matrix")
                #print(lom[i].member_matrix())
                #print("lom[j] stiffness matrix")
                #print(lom[j].member_matrix())
                start = lom[i].start_node_dof[0]
                stop = lom[i].start_node_dof[-1]
                start = int(start)
                stop = int(stop)
                gsm[start - 1:stop, start - 1:stop] += lom[i].member_matrix()[0:3, 0:3]
                #print("GSM after iteration")
                #print(gsm)
                #print("----------------------------------")
    support_dof = get_support_dof(lom)
    for member in lom:
        if start_dof.count(member.start_node_dof) >= 2:
            start = int(member.start_node_dof[0])
            stop = int(member.start_node_dof[-1])
            gsm[start-1:stop, start-1:stop] = magic(lom, member.start_node_dof)
    for dof in support_dof:
        for member in lom:
            if member.end_node_dof == dof:
                start = dof[0]
                stop = dof[-1]
                start = int(start)
                stop = int(stop)
                gsm[start - 1:stop, start - 1:stop] += member.member_matrix()[3:, 3:]  # have to fix, it's hardcoded
    start = max(end_dof)[0] - 1
    stop = max(end_dof)[-1]
    start = int(start)
    stop = int(stop)
    gsm[start:stop, start:stop] += lom[0].member_matrix()[0:3, 0:3] # have to fix, it's hardcoded
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

    for i in range(len(lom)):
        start_col = int(lom[i].end_node_dof[0])
        stop_col =  int(lom[i].end_node_dof[-1])
        start_row = int(lom[i].start_node_dof[0])
        stop_row =  int(lom[i].start_node_dof[-1])
        gsm[start_row-1: stop_row, start_col-1:stop_col] += lom[i].member_matrix()[0:3, 3:]
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
    Qk = np.zeros((num_joints*3)) #Dk = np.array((num_joints*3, 1)) not necessary for now since Dk will always just be zeros if there is no settlement etc.
    for load in loads:
        Qk[int(load.dof) - 1] -= -int(load.magnitude)
    KFF = gsm[0:num_joints*3, 0:num_joints*3]
    KFF_inv = np.linalg.inv(KFF)
    displacements = np.matmul(KFF_inv, Qk) #* 1000
    #print(["{:0.7f}".format(x) for x in displacements])
    return displacements


def compute_forces(lon, loads, gsm):
    num_joints = 0
    for node in lon:
        if node.has_support == "False":
            num_joints += 1
    KEF = gsm[num_joints*3:, 0:num_joints*3]
    Du = compute_displacements(lon, loads, gsm)
    return np.matmul(KEF, Du)

def show_loads(lol):
    for load in lol:
        if load.orientation == "Horizontal":
            x = nodes[load.node-1].x
            y = nodes[load.node-1].y
            plt.arrow(x, y, 1.4, 0, length_includes_head = True, color="red", head_width = 0.25, head_length = 0.35)


#fig = plt.figure()
#x_val = []
#y_val = []
#
#if len(nodes) > 0:
#    for k in range(len(nodes)):
#        x_val += [nodes[k].x]
#        y_val += [nodes[k].y]
#    ax = fig.add_subplot(111)
#    ax.set_aspect(1)
#    plt.plot(x_val, y_val, color = 'k')
#
def show_structure(lon, lom, lol):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    x_val = []
    y_val = []

    for i in range(len(lom)):
        ax.plot([lom[i].start_x, lom[i].end_x], [lom[i].start_y, lom[i].end_y], 'k-')
    for j in range(len(lon)):
        if lon[j].has_support == "True":
            ax.plot([lon[j].x - 0.5, lon[j].x + 0.5], [lon[j].y, lon[j].y], color = 'k')
    for k in range(len(lon)):
        x_val += [lon[k].x]
        y_val += [lon[k].y]



    if len(nodes) > 0:
        ax.set(xlim=(min(x_val) - 5, 5 + max(x_val)), ylim=(min(y_val) - 5, 5 + max(y_val)))


    st.pyplot(fig)

def show_loads(lol):
    fig = plt.figure()
    for load in lol:
        if load.orientation == "Horizontal":
            x = nodes[load.node-1].x
            y = nodes[load.node-1].y
            plt.arrow(x, y, 1.4, 0, length_includes_head = True, color="red", head_width = 0.25, head_length = 0.35)
    st.pyplot(fig)

def global_displacement_matrix(lom, lon, Du):
    gdm = Du
    for node in lon:
        if node.has_support == "True":
            gdm = np.append(gdm, [0, 0, 0])
    return gdm

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


#show_structure(nodes,members,loads)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)



answer_stiffness_matrices = st.checkbox("Show member stiffness matrices")

if answer_stiffness_matrices and len(members) >= 1:
    for i in range(len(members)):
        df = pd.DataFrame(members[i].member_matrix())
        st.write("Member #" + str(i + 1) + " - Stiffness Matrix")
        st.table(df.round(1))
    df = global_stiffness_matrix_bwd_pass(members)
    st.write("Structural System Stiffness Matrix")
    st.table(df)

if len(nodes) >= 2 and len(members) >= 1 and members[0].member_length > 0:
    gsm = global_stiffness_matrix_bwd_pass(members)

col5, col6 = st.columns(2)

with col5:
    number2 = math.floor(st.number_input('Click "+" to add nodal loads', min_value=0, value=0, step=1, key="loads"))

with col4:
    st.empty()

add_load(number2)



x_val = []
y_val = []

for i in range(len(members)):
    ax.plot([members[i].start_x, members[i].end_x], [members[i].start_y, members[i].end_y], 'k-')
for j in range(len(nodes)):
    if nodes[j].has_support == "True":
        ax.plot([nodes[j].x - 0.5, nodes[j].x + 0.5], [nodes[j].y, nodes[j].y], color = 'k')
for k in range(len(nodes)):
    x_val += [nodes[k].x]
    y_val += [nodes[k].y]



if len(nodes) > 0:
    ax.set(xlim=(min(x_val) - 5, 5 + max(x_val)), ylim=(min(y_val) - 5, 5 + max(y_val)))




if len(loads) > 0:
    for i in range(len(loads)):
        if loads[i].orientation == 'Horizontal':
            x = nodes[loads[i].node - 1].x
            y = nodes[loads[i].node - 1].y
            if x - 1 < 0:
                plt.arrow(-2, y, 2, 0, length_includes_head = True, color="red", head_width = 0.25, head_length = 0.35)
            else:
                plt.arrow(x, y, 2, 0, length_includes_head=True, color="red", head_width=0.25, head_length=0.35)
        elif loads[i].orientation == 'Vertical':
            x = nodes[loads[i].node - 1].x
            y = nodes[loads[i].node - 1].y
           # print("Vertical", x, y)
            plt.arrow(x, y * 1.5, 0, -y * 1.5 + y, length_includes_head=True, color="red", head_width=0.25, head_length=0.35)
        elif loads[i].orientation == 'Moment':
            x = nodes[loads[i].node - 1].x
            y = nodes[loads[i].node - 1].y
            style = "Simple, tail_width=0.5, head_width=4, head_length=4"
            kw = dict(arrowstyle=style, color="r")
            a1 = patches.FancyArrowPatch((x - 0.75, y), (x, y), **kw)
            a2 = patches.FancyArrowPatch((x, y), (x + 0.75, y), **kw)
            a3 = patches.FancyArrowPatch((x - 0.75, y), (x + 0.75, y), connectionstyle="arc3, rad = 0.5", **kw)

            plt.gca().add_patch(a3)


st.pyplot(fig)

answer_compute_forces = st.button("Solve system")



if answer_compute_forces:
    displacements = compute_displacements(nodes, loads, gsm)
    forces = compute_forces(nodes, loads, gsm)
    #st.write("Displacement Matrix")
    #st.write(displacements)
    #st.write("Forces")
    #st.write(forces)
    q = []
    # BMD and SFD
    T = np.array([[np.cos(members[i].theta * np.pi / 180), np.sin(members[i].theta * np.pi / 180), 0, 0, 0, 0],
                  [-np.sin(members[i].theta * np.pi / 180), np.cos(members[i].theta * np.pi / 180), 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, np.cos(members[i].theta * np.pi / 180), np.sin(members[i].theta * np.pi / 180), 0],
                  [0, 0, 0, -np.sin(members[i].theta * np.pi / 180), np.cos(members[i].theta * np.pi / 180), 0],
                  [0, 0, 0, 0, 0, 1]])
    #print(global_displacement_matrix(members,nodes, displacements))
    global_d = global_displacement_matrix(members, nodes, displacements)
    for i in range(len(members)):
        k = members[i].member_matrix()
        idx1 = int(members[i].x_start_dof)-1
        idx2 = int(members[i].y_start_dof)-1
        idx3 = int(members[i].z_start_dof)-1
        idx4 = int(members[i].x_end_dof)-1
        idx5 = int(members[i].y_end_dof)-1
        idx6 = int(members[i].z_end_dof)-1

        D = np.array([[global_d[idx1]],
                      [global_d[idx2]],
                      [global_d[idx3]],
                      [global_d[idx4]],
                      [global_d[idx5]],
                      [global_d[idx6]]])
        res = np.matmul(k, D)
        q += [np.matmul(T, res)]

    for i in range(len(q)):
        q[i] = q[i]*-1

    #print(displacements)
    #print(q)

    #for i in range(len(members)):
    #    x = 0
    #    y = 1
    #    z = 2
    #    if nodes[members[i].start_node].has_support == "True":
    #        D = np.array([[0], [0], [0], [displacements[x]], [displacements[y]], [displacements[z]]])
    #        x += 3
    #        y += 3
    #        z += 3
    #        res = np.matmul(members[i].member_matrix(), D)
    #        q += np.matmul(T, res)
    #    elif nodes[members[i].end_node].has_support == "True":
    #        D = np.array([displacements[x]], [displacements[y]], [displacements[z]], [0], [0], [0])
    #        x += 3
    #        y += 3
    #        z += 3
    #        res = np.matmul(members[i].member_matrix(), D)
    #        q += np.matmul(T, res)
    #    else:
    #        D = np.array()

    #for i in range(len(members)):

    #    res = np.matmul(members[i].member_matrix(), displacements)
    #    q[i] = np.matmul(T, res)
    #    print(q)

    col7, col8 = st.columns(2)
    with col7:
        st.write("Forces [kN]")
        st.write(forces)
    with col8:
        st.write("Displacements [mm/rad]")
        st.write(displacements)
    for i in range(len(q)):
        st.write("End moments")
        st.write(q[i])

bmd = st.checkbox("Show BMD")
if bmd:
    for i in range(1, len(members) + 1):
        if nodes[int(members[i-1].end_node)-1].y > nodes[int(members[i-1].start_node)-1].y:
            start_x = nodes[int(members[i-1].start_node)-1].x + (q[i-1][2]/abs(q[i-1][2])) * 1.2
            start_y = nodes[int(members[i-1].start_node)-1].y
            end_x = nodes[int(members[i-1].end_node)-1].x + (-1*q[i-1][5]/abs(q[i-1][5])) * 1.2
            end_y = nodes[int(members[i-1].end_node)-1].y
            ax.plot([start_x, end_x], [start_y, end_y], color='red')
            ax.plot([nodes[int(members[i - 1].end_node) - 1].x, start_x],
                    [nodes[int(members[i - 1].start_node) - 1].y, nodes[int(members[i - 1].start_node) - 1].y],
                    color='red')

            ax.plot([end_x, nodes[int(members[i-1].end_node)-1].x], [nodes[int(members[i-1].end_node)-1].y,nodes[int(members[i-1].end_node)-1].y], color="red")
        elif nodes[int(members[i-1].end_node)-1].y < nodes[int(members[i-1].start_node)-1].y:
            start_x = nodes[int(members[i-1].end_node)-1].x + (-1*q[i-1][5]/abs(q[i-1][5])) * 1.2
            start_y = nodes[int(members[i-1].start_node)-1].y
            end_x = nodes[int(members[i-1].start_node)-1].x + (q[i-1][2]/abs(q[i-1][2])) * 1.2
            end_y = nodes[int(members[i-1].end_node)-1].y
            ax.plot([start_x, end_x], [start_y, end_y], color='red')
            ax.plot([nodes[int(members[i-1].end_node)-1].x, start_x], [nodes[int(members[i-1].start_node)-1].y, nodes[int(members[i-1].start_node)-1].y], color='red')
            ax.plot([end_x, nodes[int(members[i - 1].end_node) - 1].x],
                    [nodes[int(members[i - 1].end_node) - 1].y, nodes[int(members[i - 1].end_node) - 1].y], color="red")
        elif nodes[int(members[i-1].end_node)-1].y == nodes[int(members[i-1].start_node)-1].y:
            start_x = nodes[int(members[i - 1].start_node) - 1].x
            start_y = nodes[int(members[i - 1].end_node) - 1].y + (-1 * q[i - 1][5] / abs(q[i - 1][5])) * 1.2
            end_x = nodes[int(members[i - 1].end_node) - 1].x
            end_y = nodes[int(members[i - 1].start_node) - 1].y + (q[i - 1][2] / abs(q[i - 1][2])) * 1.2
            ax.plot([start_x, end_x], [start_y, end_y], color='red')
            ax.plot([start_x, start_x], [nodes[int(members[i - 1].end_node) - 1].y, start_y], color='red')
            ax.plot([end_x, end_x], [nodes[int(members[i - 1].end_node) - 1].y,end_y], color='red')
    #for i in range(1, len(members) + 1): # nodes[i-1], nodes[i]
    #    if nodes[i].y > nodes[i-1].y: # this means member is going upwards (i.e., vertical)
    #        start_x = nodes[i-1].x  + (q[i-1][2]/abs(q[i-1][2])) * 1.25
    #        start_y = nodes[i-1].y
    #        end_x = nodes[i].x + (q[i][2]/abs(q[i][2])) * 1.25
    #        end_y = nodes[i].y
    #        ax.plot([start_x, end_x], [start_y, end_y], color = 'red')
    #    elif nodes[i].y < nodes[i-1].y: # this means member is going upwards (i.e., vertical)
    #        start_x = nodes[i-1].x  + (q[i-1][2]/abs(q[i-1][2])) * 1.25
    #        start_y = nodes[i-1].y
    #        end_x = nodes[i].x + (q[i][2]/abs(q[i][2])) * 1.25
    #        end_y = nodes[i].y
    #        ax.plot([start_x, end_x], [start_y, end_y], color = 'red')
    #    elif nodes[i].y == nodes[i-1].y: # straight member (i.e., horizontal)
    #        start_x = nodes[i-1].x
    #        #start_y = nodes[i-1].y + (q[i-1][2]/abs)
    #        end_x = nodes[i].x
    #        end_y = 2
    st.pyplot(fig)
