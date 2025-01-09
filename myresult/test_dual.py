import argparse
import os
import torch
from torch.utils.data import DataLoader

from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf,compute_eer_A1_A20
from tqdm import tqdm
from evaluate_EER_2015 import compute_EER
from model import Model
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_eval,Dataset_ASVspoof2021_eval,genSpoof_list2021DF,genSpoof_list2019,genSpoof_list2015,Dataset_ASVspoof2015_eval,genSpoof_list_wild,Dataset_wild_eval
from evaluate_2021_LA import eval_to_score_file
from evaluate_2021_DF import eval_to_score_fileDF



def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)



def test_model_2019(args,model_path,output_dir):
    model = Model(args, args.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(args.device)
    print('nb_params:', nb_params)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
    print('Model loaded : {}'.format(model_path))
    print("2019LA......")
    file_eval,tags,lables = genSpoof_list2019(
        dir_meta='E:/datas/ASVspoof_2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2019_eval(list_IDs=file_eval, base_dir='E:/datas/ASVspoof_2019/LA/ASVspoof2019_LA_eval/',tags=tags,lables=lables)

    produce_evaluation_file2019(eval_set, model, args.device, output_dir + '/' + args.epoch)

    compute_eer_and_tdcf(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt", '')
    compute_eer_A1_A20(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt", output_dir + '/' + args.epoch)
    return 0

def test(args):


    model_path = args.model_dir+ '/epoch_' + args.epoch + ".pth"
    output_dir_2021LA = args.output_dir+'/2021LA'+args.cla
    output_dir_2021DF = args.output_dir + '/2021DF' + args.cla
    output_dir_2019 = args.output_dir + '/2019' + args.cla
    output_dir_2015 = args.output_dir + '/2015' + args.cla
    output_dir_wild = args.output_dir + '/wild' + args.cla

    test_model_2015(args, model_path,output_dir_2015)
    test_model_2019(args, model_path,output_dir_2019)
    test_model_2021(args, model_path,output_dir_2021LA)
    test_model_2021DF(args, model_path,output_dir_2021DF)
    test_model_wild(args, model_path,output_dir_wild)



def test_model_2015(args,model_path,output_dir):
    model = Model(args, args.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(args.device)
    print('nb_params:', nb_params)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
    print('Model loaded : {}'.format(model_path))
    print("2015......")
    name,file_eval, lables = genSpoof_list2015(
        dir_meta='E:/datas/ASVspoof_2015/CM_protocol/cm_evaluation.ndx.txt',
        is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2015_eval(list_IDs=file_eval,
                                         base_dir='E:/datas/ASVspoof_2015/', lables=lables,list_files=name)

    produce_evaluation_file2015(eval_set, model, args.device, output_dir + '/' + args.epoch)

    compute_EER(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt")

    return 0

def produce_evaluation_file(dataset, model, device, save_path):
    check_and_create_path(save_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()


    for batch_x, utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(os.path.join(save_path, 'checkpoint_cm_score.txt'), 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f,cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))

def produce_evaluation_file2019(dataset, model, device, save_path):
    check_and_create_path(save_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()


    for batch_x, utt_id ,tag,lable in tqdm(data_loader):
        fname_list = []
        score_list = []
        tag_list = []
        lable_list = []
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        tag_list.extend(tag)
        lable_list.extend(lable)

        with open(os.path.join(save_path, 'checkpoint_cm_score.txt'), 'a+') as fh:
            for f,t,l, cm in zip(fname_list,tag_list,lable_list, score_list):
                fh.write('{} {} {} {}\n'.format(f,t,l,cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))

def produce_evaluation_file2015(dataset, model, device, save_path):
    check_and_create_path(save_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()


    for batch_x, utt_id ,lable in tqdm(data_loader):
        fname_list = []
        score_list = []
        lable_list = []
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lable_list.extend(lable)

        with open(os.path.join(save_path, 'checkpoint_cm_score.txt'), 'a+') as fh:
            for f,l, cm in zip(fname_list,lable_list, score_list):
                fh.write('{} {} {}\n'.format(f,l,cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))

def produce_evaluation_file_wild(dataset, model, device, save_path):
    check_and_create_path(save_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()


    for batch_x, utt_id ,lable in tqdm(data_loader):
        fname_list = []
        score_list = []
        lable_list = []
        batch_x = batch_x.to(device)  # (10,64600)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lable_list.extend(lable)

        with open(os.path.join(save_path, 'checkpoint_cm_score.txt'), 'a+') as fh:
            for f,l, cm in zip(fname_list,lable_list, score_list):
                fh.write('{} {} {}\n'.format(f,l,cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))


def test_model_2021(args,model_path,output_dir):
    model = Model(args, args.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(args.device)
    print('nb_params:', nb_params)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
    print('Model loaded : {}'.format(model_path))
    print("2021LA......")
    file_eval = genSpoof_list(
        dir_meta='E:/datas/keys/LA/CM/trial_metadata.txt',
        is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='E:/datas/datas/ASVspoof2021_LA_eval/')

    produce_evaluation_file(eval_set, model, args.device, output_dir+'/'+args.epoch)
    eval_to_score_file(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt",
                       "E:/datas/keys/LA/CM/trial_metadata.txt",
                       "E:/datas/keys/LA/ASV/trial_metadata.txt",
                       "E:/datas/keys/LA/ASV/ASVTorch_Kaldi/score.txt", A07_A20=True)

    return 0

def test_model_wild(args,model_path,output_dir):
    model = Model(args, args.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(args.device)
    print('nb_params:', nb_params)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
    print('Model loaded : {}'.format(model_path))
    print("in the wild......")
    file_eval, lables = genSpoof_list_wild(
        dir_meta='E:/datas/release_in_the_wild/meta.csv',
        is_train=False, is_eval=True)
    eval_set = Dataset_wild_eval(list_IDs=file_eval,base_dir='E:/datas/release_in_the_wild/', lables=lables)

    produce_evaluation_file_wild(eval_set, model, args.device, output_dir + '/' + args.epoch)

    compute_EER(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt")

    return 0


def test_model_2021DF(args,model_path,output_dir):
    model = Model(args, args.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(args.device)
    print('nb_params:', nb_params)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
    print('Model loaded : {}'.format(model_path))
    print("2021DF......")
    file_eval = genSpoof_list2021DF(
        dir_meta='E:/datas/keys/DF/CM/trial_metadata.txt',
        is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='E:/datas/ASVspoof2021_DF_eval/')

    produce_evaluation_file(eval_set, model, args.device, output_dir + '/' + args.epoch)
    eval_to_score_fileDF(output_dir + '/' + args.epoch + "/checkpoint_cm_score.txt",
                       "E:/datas/keys/DF/CM/trial_metadata.txt",)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, default="../model/")
    parser.add_argument('-o', '--output_dir', type=str, default="./jj+BLDAM-PRW")
    parser.add_argument('-c', '--cla', type=str, default="")
    parser.add_argument('-e', '--epoch', type=str,default="90")
    parser.add_argument('--algo', type=int, default=5,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                              5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)
    test(args)








