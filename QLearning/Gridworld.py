#主要是基于DeepRL-Agents的开源实现
import numpy as np    
import random   
import itertools  
import scipy.misc   
import matlplotlib.pyplot as plt   
import tensorflow as tf   
import os   

#创建的是关于环境物体对象的class
class gameob():
    def  __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = corrdinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name   

#之后我们需要创建的是环境的class 
class gameEnv():
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size  
        self.actions = 4   
        self.objects = []
        a = self.reset()
        plt.imshow(a, interpolation = "nearest")
        #将所有的物体添加到object中
    def reset(self):
        self.objects = []
        hero = gameob(self.newPosition(),1,1,2,None, 'hero')
        self.objects.append(hero)
        goal = gameob(self.newPosition(),1,1,1,'goal')
        self.objects.append(goal)
        hole = gameob(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        goal2 = gameob(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal2)
        hole2 = gameob(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        goal3 = gameob(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal3)
        goal4 = gameob(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state 
        return state 
    def movechar(self, direction):
        hero = self.objects[0]
        heroX = hero.x  
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y-=1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y +=1
        if direction == 2 and hero.x >= 1:
            hero.x -=1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x +=1
        self.objects[0] = hero
    
    