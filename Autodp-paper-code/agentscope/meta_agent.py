# -*- coding: utf-8 -*-
"""A simple example for conversation between user and assistant agent."""
from tools import load_txt
import argparse
import pdb
import subprocess
import pickle
from openai import OpenAI
from data_process_action_choice import Action
import json

def obverse_team(chat_response):
    text = chat_response.split('</think>')[1].strip()

    length = len(text.split('**###组合'))

    if length == 5:
        team1 = text.split('**###组合[1]###**')[1].split('**###组合[2]###**')[0].strip().replace('●','').replace(' ','')
        team2 = text.split('**###组合[2]###**')[1].split('**###组合[3]###**')[0].strip().replace('●','').replace(' ','')
        team3 = text.split('**###组合[3]###**')[1].split('**###组合[4]###**')[0].strip().replace('●','').replace(' ','')
        team4 = text.split('**###组合[4]###**')[1].split('**###不同组合的理由')[0].strip().replace('●','').replace(' ','')

    if length == 4:
        team1 = text.split('**###组合[1]###**')[1].split('**###组合[2]###**')[0].strip().replace('●','').replace(' ','')
        team2 = text.split('**###组合[2]###**')[1].split('**###组合[3]###**')[0].strip().replace('●','').replace(' ','')
        team3 = text.split('**###组合[3]###**')[1].split('**###不同组合的理由')[0].strip().replace('●','').replace(' ','')
        team4 = None
    
    if length == 3:
        team1 = text.split('**###组合[1]###**')[1].split('**###组合[2]###**')[0].strip().replace('●','').replace(' ','')
        team2 = text.split('**###组合[2]###**')[1].split('**###不同组合的理由')[0].strip().replace('●','').replace(' ','')
        team3 = None
        team4 = None
    
    if length == 2:
        team1 = text.split('**###组合[1]###**')[1].split('**###不同组合的理由')[0].strip().replace('●','').replace(' ','')
        team2 = None
        team3 = None
        team4 = None
    
    return team1, team2, team3, team4


if __name__ == "__main__":
    """A basic conversation demo"""

    score_record = {}

    prompt4 = "1. **###组合[1]###**\n- 反馈得分：{}\n2. **###组合[2]###**\n- 反馈得分：{}\n3. **###组合[3]###**\n- 反馈得分：{}\n 4. **###组合[4]###**\n- 反馈得分：{}\n\n"
    prompt3 = "1. **###组合[1]###**\n- 反馈得分：{}\n2. **###组合[2]###**\n- 反馈得分：{}\n3. **###组合[3]###**\n- 反馈得分：{}\n\n"
    prompt2 = "1. **###组合[1]###**\n- 反馈得分：{}\n2. **###组合[2]###**\n- 反馈得分：{}\n\n"
    prompt1 = "1. **###组合[1]###**\n- 反馈得分：{}\n\n"

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    with open("meta_agent_instruct/round1_instruction.txt",'r',encoding='utf-8') as fr:
        round1_prompt = fr.readlines()
        round1_prompt = ''.join(round1_prompt)
    
    message=[{"role": "user", "content": round1_prompt}]

    round_num = 1

    while round_num <= 6:

        chat_response = client.chat.completions.create(
            model="/mnt/data3/nianke_multi_agent/model/Qwen3-32B",
            messages=message,
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
        )
        chat_response = chat_response.choices[0].message.content

        text = chat_response.split('</think>')[1]
        print("Chat response:", text)

        if '【最佳团队】' in text.split('###不同组合的理由')[0]:
            print("########################################")
            print("................最佳策略已选择.............")
            print("########################################")
            break

        if '【原始数据无须任何处理】' in text.split('###')[1].split('###')[0]:
            print("########################################")
            print("................原始数据无须优化.............")
            print("########################################")
            break

        ###########获取每组的反馈得分###########
        team1, team2, team3, team4 = obverse_team(chat_response)
        action = Action(team1, team2, team3, team4)
        team1_score, team2_score, team3_score, team4_score = action.forward(score_record)

        dict = {}
        dict[f'第{round_num}轮文本'] = text
        dict[f'第{round_num}轮团队得分'] = [team1_score, team2_score, team3_score, team4_score]
        with open('result_team.jsonl', 'a', encoding='utf-8') as file:  
            json_line = json.dumps(dict, ensure_ascii=False)
            file.write(json_line + '\n')
    

        ###########需要记录每组中每个策略的得分，因为后面可能回重复使用，这样就不要消耗时间重新处理数据######
        if team1 != None and team1 not in score_record.keys():
            score_record[team1] = team1_score
        if team2 != None and team2 not in score_record.keys():
            score_record[team2] = team2_score
        if team3 != None and team3 not in score_record.keys():
            score_record[team3] = team3_score
        if team4 != None and team4 not in score_record.keys():
            score_record[team4] = team4_score
        
        ###############生成第二轮prompt#############
        
        num = [team1, team2, team3, team4]
        length = len([ans for ans in num if ans == None])
        
        if length == 0:
            round2_prompt = "第{}轮的不同的团队组合的反馈得分为：\n{} ### 第{}轮\n现在请执行第{}步，评估组合的反馈得分并调整组合以及工作顺序允许改变组合数量。".format(round_num, prompt4.format(team1_score, team2_score, team3_score, team4_score), round_num+1, round_num+1)

        if length == 1:
            round2_prompt = "第{}轮的不同的团队组合的反馈得分为：\n{} ### 第{}轮\n现在请执行第{}步，评估组合的反馈得分并调整组合以及工作顺序允许改变组合数量。".format(round_num, prompt3.format(team1_score, team2_score, team3_score),round_num+1, round_num+1)

        if length == 2:
            round2_prompt = "第{}轮的不同的团队组合的反馈得分为：\n{} ### 第{}轮\n现在请执行第{}步，评估组合的反馈得分并调整组合以及工作顺序允许改变组合数量。".format(round_num, prompt2.format(team1_score, team2_score), round_num+1, round_num+1)
        
        if length == 3:
            round2_prompt = "第{}轮的不同的团队组合的反馈得分为：\n{} ### 第{}轮\n现在请执行第{}步，评估组合的反馈得分并调整组合以及工作顺序允许改变组合数量。".format(round_num, prompt1.format(team1_score), round_num+1, round_num+1)

        #############多轮对话形式(将之前对话变成历史对话)#########
        dict1 = {}
        dict1['role'] = 'assistant'
        dict1['content'] = chat_response

        dict2 = {}
        dict2['role'] = 'user'
        dict2['content'] = round2_prompt
        message.append(dict1)
        message.append(dict2)

        round_num += 1
        