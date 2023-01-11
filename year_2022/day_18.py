import logging
from utils import load_file
import re
import numpy as np
from tqdm import tqdm
from collections import Counter

# Total surface area of connected voxels
class Cube:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.all_verticies()
        self.all_faces()

    def all_verticies(self):
        # Get all 8 verticies of the cube
        self.verticies = dict()
        self.verticies[0] = (self.x, self.y, self.z)
        self.verticies[1] = (self.x, self.y + 1, self.z)
        self.verticies[2] = (self.x, self.y, self.z + 1)
        self.verticies[3] = (self.x, self.y + 1, self.z + 1)
        self.verticies[4] = (self.x + 1, self.y + 1, self.z)
        self.verticies[5] = (self.x + 1, self.y, self.z + 1)
        self.verticies[6] = (self.x + 1, self.y, self.z)
        self.verticies[7] = (self.x + 1, self.y + 1, self.z + 1)
    
    def all_faces(self):
        self.faces = dict()

        self.faces[0] = tuple(sorted((self.verticies[0], self.verticies[1], self.verticies[2], self.verticies[3])))
        self.faces[1] = tuple(sorted((self.verticies[2], self.verticies[5], self.verticies[3], self.verticies[7])))
        self.faces[2] = tuple(sorted((self.verticies[5], self.verticies[7], self.verticies[4], self.verticies[6])))
        self.faces[3] = tuple(sorted((self.verticies[6], self.verticies[4], self.verticies[1], self.verticies[0])))
        self.faces[4] = tuple(sorted((self.verticies[1], self.verticies[3], self.verticies[4], self.verticies[7])))
        self.faces[5] = tuple(sorted((self.verticies[0], self.verticies[2], self.verticies[6], self.verticies[5])))
        

def combine_face(inputs):
    faces = []
    for coord in inputs: 
        cube = Cube(coord[0], coord[1], coord[2])
        all_faces = [cube.faces[i] for i in range(6)]
        faces.extend(all_faces)
    counts = Counter(faces)
    num_exposed_faces = sum([x for x in counts.values() if x == 1])
    logging.info(f"Part 1: {num_exposed_faces}")
    return num_exposed_faces
    
    

def part_1(inputs):
    # import ipdb; ipdb.set_trace()
    inputs = [tuple(int(x) for x in line.split(",")) for line in inputs]
    res = combine_face(inputs)
    return res

def part_2(inputs):
    inputs = [tuple(int(x) for x in line.split(",")) for line in inputs]
    faces = []
    for coord in inputs: 
        cube = Cube(coord[0], coord[1], coord[2])
        all_faces = [cube.faces[i] for i in range(6)]
        faces.extend(all_faces)
    out = [item for sublist in faces for item in sublist]
    x = [tup[0] for tup in out]
    y = [tup[1] for tup in out]
    z = [tup[2] for tup in out]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    min_z = min(z)
    max_z = max(z)
    # These are all 0 - 20 it turns out 
    # How do we get the actual outer faces with the way I've defined them...
    # Best solutions look to be 
    
    pass

def control():
    inputs = load_file("year_2022/data/data_18.txt")
    part_1(inputs)
    part_2(inputs)