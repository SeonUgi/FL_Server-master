import copy

import numpy as np

import logging
logger = logging.getLogger(__name__)

class FederatedServer:
    
    max_count = 2
    global_weight = None
    local_weights = []
    global_estimation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    local_estimations = []
    current_count = 0
    current_round = 0

    def __init__(self):
        print("Federated init")

    #Update weights, round, client_count
    @classmethod
    def update(cls, local_weight):
        weight_list = []
        for i in range(len(local_weight)): #range(4):  #todo check layer 갯수
            temp = np.array(local_weight[i])
            weight_list.append(temp)

        cls.current_count += 1
        cls.local_weights.append(weight_list)
        
        #print("update count : {}, {}".format(cls.current_count, len(cls.local_weights)))
        if cls.current_count == cls.max_count:
            cls.avg()
            cls.current_count = 0
            cls.current_round += 1
            logger.info("----------------------------------------")
            logger.info("current round : {}".format(cls.current_round))
            logger.info("----------------------------------------")

    # Update statistical estimation
    @classmethod
    def update_estimation(cls, local_estimations):
        estimation_list = []
        for i in range(len(local_estimations)): #range(4):  #todo check layer 갯수
            temp = np.array(local_estimations[i])
            estimation_list.append(temp)

        cls.local_estimations.append(estimation_list)

    @classmethod
    def est(cls):
        temp_list = []

        temp_estimation = cls.local_estimations.pop()   #   weight의 shape를 모르므로, 하나를 꺼내어 사용

        for i in range(len(temp_estimation)):  #todo check layer 갯수
            temp = np.array(temp_estimation[i])
            temp_list.append(temp)

        temp_list = np.array(temp_list)
        
        for i in range(len(cls.local_estimations)):
            for j in range(len(cls.local_estimations[i])):
                temp = np.array(cls.local_estimations[i][j])
                temp_list[j] += temp


        cls.global_estimation = np.divide(temp_list, cls.max_count)
        cls.local_estimations = []  #   global weight average 이후 다음 라운드를 위해 이전의 local weight 리스트 초기화

    @classmethod
    def avg(cls):
        temp_list = []

        temp_weight = cls.local_weights.pop()   #   weight의 shape를 모르므로, 하나를 꺼내어 사용

        for i in range(len(temp_weight)):  #todo check layer 갯수
            temp = np.array(temp_weight[i])
            temp_list.append(temp)

        temp_list = np.array(temp_list)
        
        for i in range(len(cls.local_weights)):
            for j in range(len(cls.local_weights[i])):
                temp = np.array(cls.local_weights[i][j])
                temp_list[j] += temp


        cls.global_weight = np.divide(temp_list, cls.max_count)
        cls.local_weights = []  #   global weight average 이후 다음 라운드를 위해 이전의 local weight 리스트 초기화

    @classmethod
    def get_avg(cls):
        return cls.global_weight

    @classmethod
    def get_est(cls):
        return cls.global_estimation

    @classmethod
    def get_current_count(cls):
        return cls.current_count

    @classmethod
    def get_current_round(cls):
        return cls.current_round

    @classmethod
    def set_client_count(cls, count):
        cls.max_count = count

    @classmethod
    def get_client_count(cls):
        return cls.max_count

    @classmethod
    def reset_parm(cls):
        cls.max_count = 3
        cls.global_weight = None
        cls.global_estimation = None
        cls.local_estimations = []
        cls.local_weights = []
        cls.current_count = 0
        cls.current_round = 0