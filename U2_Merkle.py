# adapted from droid76's code
from pympler.asizeof import asizeof

def toHash(value):
    return str(hash(str(value)))

def concat(a, b):
    return a + b
    
class BogusStringTemplate:
    def __init__(self, string_):
        self.string = string_

    def write(self, *argv):
        for x in argv:
            self.string += str(x)

class MerkleTreeNode:
    def __init__(self,value):
        self.left = None
        self.right = None
        self.value = value
        self.hashValue = toHash(value)

class Tree:
    def __init__(self, summary, nodes=[]):
        self.nodes = nodes
        self.summary = summary

    def __str__(self):
        return self.summary

def buildTree(leaves):
    nodes = []
    summary = BogusStringTemplate("")
    f = summary
    for i in leaves:
        nodes.append(MerkleTreeNode(i))

    while len(nodes)!=1:
        temp = []
        for i in range(0,len(nodes),2):
            node1 = nodes[i]
            if i+1 < len(nodes):
                node2 = nodes[i+1]
            else:
                temp.append(nodes[i])
                break
            f.write("Left child : ", node1.value + " | Hash : " + node1.hashValue +" \n")
            f.write("Right child : ", node2.value + " | Hash : " + node2.hashValue +" \n")
            concatenatedHash = node1.hashValue + node2.hashValue
            parent = MerkleTreeNode(concatenatedHash)
            parent.left = node1
            parent.right = node2
            f.write("Parent(concatenation of "+ node1.value + " and " + node2.value + ") : " +parent.value + " | Hash : " + parent.hashValue +" \n")
            temp.append(parent)
        nodes = temp
    return Tree(f.string, nodes[0])

def isIdentical(treeA, treeB):
    pass

def checkConsistency(leaves1,leaves2):
    i=0
    while i<len(leaves1):
        if leaves1[i]!=leaves2[i]:
            break
        i+=1
    if i < len(leaves1):
        return []
    s = ""
    s += "Merkle Tree 1 \n"
    t1 = buildTree(leaves1)
    tree1 = t1.nodes
    s += str(t1)
    s += "\n\n"
    s += "Merkle Tree 2 \n"
    t2 = buildTree(leaves2)
    tree2 = t2.nodes
    s += str(t2)
    op = []
    op.append(tree1.hashValue)
    data = s.split('\n')
    for i in range(len(data)):
        data[i] += '\n'
    data = data[:-1]

    tree2Index = 0
    for i in range(len(data)):
        if data[i].startswith("Merkle Tree 2"):
            tree2Index = i
    parentLines = []
    leftChildLines = []
    rightChildLines = []
    for i in range(tree2Index,len(data)):
        if data[i].startswith("Parent("):
            parentLines.append(data[i])
    
    for i in range(tree2Index,len(data)):
        if data[i].startswith("Left"):
            leftChildLines.append(data[i])

    for i in range(tree2Index,len(data)):
        if data[i].startswith("Right"):
            rightChildLines.append(data[i])  
    op = []
    flag = False
    for i in range(len(parentLines)):
        if tree1.hashValue in parentLines[i]:
            flag = True
            break
    if flag:
        values = []    
        combinedHash = ''
        lc = tree1.value
        while combinedHash != tree2.hashValue:
            for i in range(len(leftChildLines)):
                if lc in leftChildLines[i].split(" ")[-6]:
                    rc = rightChildLines[i].split(" ")[-6]
                    values.append(toHash(rc))
                    break
            combinedValue = concat(toHash(lc),toHash(rc))
            combinedHash = toHash(combinedValue)
            lc = combinedValue
            
        op.append(tree1.hashValue)
        op+=values
        op.append(tree2.hashValue)
                
    else:
        root1LeftChildValue = data[tree2Index-5].split(" ")[-6]
        root1RightChildValue = data[tree2Index-4].split(" ")[-6]
        root1RightChildSiblingValue = leaves2[leaves2.index(root1RightChildValue)+1]
        values = []
        values.append(toHash(root1LeftChildValue))
        values.append(toHash(root1RightChildValue))
        values.append(toHash(root1RightChildSiblingValue))
        root1RightChildCombinedValue = concat(toHash(root1RightChildValue),toHash(root1RightChildSiblingValue))        
        combinedHash = ''
        lc = root1LeftChildValue
        rc = root1RightChildCombinedValue
        while combinedHash != tree2.hashValue:
            combinedValue = concat(toHash(lc),toHash(rc))
            combinedHash = toHash(combinedValue)
            lc = combinedValue
            for i in range(len(leftChildLines)):
                if lc in leftChildLines[i].split(" ")[-6]:
                    rc = rightChildLines[i].split(" ")[-6]
                    values.append(toHash(rc))
                    break
            
        op.append(tree1.hashValue)
        op+=values
        op.append(tree2.hashValue)
                
    return op

def checkConsistencyWrap(tree1, tree2):
    # true if identical, false if different
    try:
        checkConsistency(tree1,tree2)
        return False
    except:
        return True

def isInTree(inputString,tree):
    tree = convertToParseable(str(buildTree(tree)).split('\n'))
    op = []
    for key,value in tree.items():
        if inputString in key:
            op.append(value)
            inputString = value
    return len(op)> 0

def convertToParseable(treestr):
    tree ={}
    for line in treestr[:-1]:
        lineArray = line.split(" ")
        if lineArray[0] == 'Parent(concatenation':
            tree[lineArray[6]] = lineArray[10]
        else:
            tree[lineArray[3]] = lineArray[7]
    return tree