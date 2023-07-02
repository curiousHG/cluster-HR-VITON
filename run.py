import subprocess
import os
pwd = os.getcwd()

# Train Condition GAN
for i in range(3):
    t =  [
        'python3', 'train_condition.py', '--cuda','True','--Ddownx2', '--Ddropout', '--lasttvonly', 
        '--interflowloss', '--occlusion', '--name', f'full_test_amd{i}', '--data_list', f'data4/train{i}.txt', 
        '--tocg_checkpoint', f'{pwd}/checkpoints/mtviton.pth',
        '--keep_step','30000','--test_data_list',f'data4/test{i}.txt','--tensorboard_count','50'
    ]
    print(' '.join(t))
    # f = open(f'tcog_output{i}.txt', "w")
    # e = open(f'tcog_error{i}.txt', "w")
    # subprocess.call(t,stdout = f,stderr = e)

# Train Generator GAN
for i in range(3):
    t = [
        'python3', 'train_generator.py','--cuda','True','--occlusion','-b','4','-j','8',
        '--name', f'cluster_gen_train{i}',  
        '--tocg_checkpoint', f'{pwd}/checkpoints/mtviton.pth',
        '--gen_checkpoint', f'{pwd}/checkpoints/gen.pth',
        '--dis_checkpoint', f'{pwd}/checkpoints/discriminator_mtviton.pth',
        '--keep_step','10000',
        '--decay_step','10000',
        '--data_list', f'data4/train{i}.txt',
        '--test_data_list',f'data4/test{i}.txt',
        '--tensorboard_count','50',
    ]
    print(' '.join(t))
    # f = open(f'tiog_output{i}.txt', "w")
    # e = open(f'tiog_error{i}.txt', "w")
    # subprocess.call(t,stdout = f,stderr = e)
    
    
