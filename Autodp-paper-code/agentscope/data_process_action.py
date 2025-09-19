from typing import Optional

import regex as re
from datetime import datetime
import argparse
import pdb
import json
from tqdm import tqdm
import subprocess
# from medical_win_loss_baichuan import Evaluation
import time
from pathlib import Path

class Action:
    """返回清洗后的数据"""

    def __init__(self,
                team1 = "数据清洗团队,数据挑选团队",
                team2 = None,
                team3 = None,
                team4 = None,              
                ):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        """
        self.team1 = team1
        self.team2 = team2
        self.team3 = team3
        self.team4 = team4

        self.data_process_team = ['数据清洗团队','数据优化团队','数据生成团队','数据挑选团队']

        self.dirty_data_sample_path = 'dq/data_sample/train_medical_sample.jsonl'

        self.data_clean_save_path = 'data_process_api/data_clean/train_medical_sample/train_medical_sample_clean.jsonl'
        self.data_optimizer_save_path = 'data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer.jsonl'
        self.data_generate_save_path = 'data_process_api/data_generate/train_medical_sample/train_medical_sample_generate.jsonl'
        self.data_select_save_path = 'data_process_api/data_select/selected_data/train_medical_sample/train_medical_sample_select.jsonl'

        self.data_process_save_path = [self.data_clean_save_path,self.data_optimizer_save_path,self.data_generate_save_path,self.data_select_save_path]

        self.script = ['data_process_api/data_clean/data_clean.py','data_process_api/data_optimize/data_optimizer.py','data_process_api/data_generate/data_generate.py','data_process_api/data_select/LESS/data_select.sh']

    
    def del_save_path(self):
        com = """
        rm data_process_api/data_clean/train_medical_sample/train_medical_sample_clean.jsonl
        rm data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer.jsonl
        rm data_process_api/data_generate/train_medical_sample/train_medical_sample_generate.jsonl
        rm data_process_api/data_select/selected_data/train_medical_sample/train_medical_sample_select.jsonl
        rm data_process_api/data_optimize/data/clean_or_dirty.pkl
        rm data_process_api/data_generate/train_medical_sample/train_medical_sample_generate_question.jsonl
        rm data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer_question.jsonl
        """
        subprocess.run(com,shell=True)
    
    def compute_stats(self, members):
        self.del_save_path()
        i = 0
        record = []
        #################数据处理阶段##############
        for member in members:
            if member not in self.data_process_team:
                print("...............组合的成员名称对不上，可能存在错误需要..........")
                pdb.set_trace()
            index = self.data_process_team.index(member)
            record.append(index)
            if i == 0:
                if index != 3:
                    subprocess.run(['python', self.script[index], '--data_path', self.dirty_data_sample_path])
                else:
                    subprocess.run(['bash', self.script[index], self.dirty_data_sample_path])             
                i = i + 1               
            else:
                if index != 3:
                    subprocess.run(['python', self.script[index], '--data_path', self.data_process_save_path[record[i-1]]])
                else:
                    subprocess.run(['bash', self.script[index], self.data_process_save_path[record[i-1]]])
                i = i + 1 
        NOW_TIME = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")
        ################模型训练###############
        if members[-1] == '数据挑选团队':
            subprocess.run(['bash', 'run_full_qwen_select.sh', self.data_process_save_path[record[-1]], NOW_TIME])
        else:
            subprocess.run(['bash', 'run_full_qwen.sh', self.data_process_save_path[record[-1]], NOW_TIME])
        time.sleep(20)
        ###############生成测试集答案############
        base_path = Path('/mnt/data3/nianke_multi_agent/model/qwen2_5_1_5b_ins/full/train_medical_sample_clean_' + NOW_TIME)
        matched_files = base_path.glob('v*/checkpoint-*')
        file_path = [ans for ans in matched_files] 
        subprocess.run(['python', 'evaluation_infer_vllm.py', '--model_path', file_path[0]])
        time.sleep(20)
        ###########计算反馈得分############
        result = subprocess.run(['python','medical_win_loss_baichuan.py'],capture_output=True,text=True)
        score = result.stdout.split('反馈得分：')[1].strip()
        print("...............反馈得分.............:",score)
        time.sleep(20)

        return score   

    def forward(self,score_record):
        # check if it's computed already
        ###################团队1#############
        if self.team1 != None:
            if self.team1 in score_record.keys():
                team1_score = score_record[self.team1]
            else:
                if ',' not in self.team1:
                    members = [self.team1]
                else:
                    members = self.team1.split(',')
                
                team1_score = self.compute_stats(members)
        else:
            team1_score = None

        ###################团队2#############
        if self.team2 != None:
            if self.team2 in score_record.keys():
                team2_score = score_record[self.team2]
            else:
                if ',' not in self.team2:
                    members = [self.team2]
                else:
                    members = self.team2.split(',')
                
                team2_score = self.compute_stats(members)
        else:
            team2_score = None

        ###################团队3#############
        if self.team3 != None:
            if self.team3 in score_record.keys():
                team3_score = score_record[self.team3]
            else:
                if ',' not in self.team3:
                    members = [self.team3]
                else:
                    members = self.team3.split(',')
                
                team3_score = self.compute_stats(members)
        else:
            team3_score = None

        ###################团队4#############
        if self.team4 != None:
            if self.team4 in score_record.keys():
                team4_score = score_record[self.team4]
            else:
                if ',' not in self.team4:
                    members = [self.team4]
                else:
                    members = self.team4.split(',')
                
                team4_score = self.compute_stats(members)
        else:
            team4_score = None

        return team1_score, team2_score, team3_score, team4_score
            

if __name__ == "__main__":

    action = Action()
    action.forward()