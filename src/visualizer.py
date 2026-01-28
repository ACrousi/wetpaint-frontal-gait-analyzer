import os, logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils as U


class Visualizer():
    def __init__(self, args):
        self.args = args
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['ntu'] = [
            'drink water 1', 'eat meal/snack 2', 'brushing teeth 3', 'brushing hair 4', 'drop 5', 'pickup 6',
            'throw 7', 'sitting down 8', 'standing up 9', 'clapping 10', 'reading 11', 'writing 12',
            'tear up paper 13', 'wear jacket 14', 'take off jacket 15', 'wear a shoe 16', 'take off a shoe 17',
            'wear on glasses 18','take off glasses 19', 'put on a hat/cap 20', 'take off a hat/cap 21', 'cheer up 22',
            'hand waving 23', 'kicking something 24', 'put/take out sth 25', 'hopping 26', 'jump up 27',
            'make a phone call 28', 'playing with a phone 29', 'typing on a keyboard 30',
            'pointing to sth with finger 31', 'taking a selfie 32', 'check time (from watch) 33',
            'rub two hands together 34', 'nod head/bow 35', 'shake head 36', 'wipe face 37', 'salute 38',
            'put the palms together 39', 'cross hands in front 40', 'sneeze/cough 41', 'staggering 42', 'falling 43',
            'touch head 44', 'touch chest 45', 'touch back 46', 'touch neck 47', 'nausea or vomiting condition 48',
            'use a fan 49', 'punching 50', 'kicking other person 51', 'pushing other person 52',
            'pat on back of other person 53', 'point finger at the other person 54', 'hugging other person 55',
            'giving sth to other person 56', 'touch other person pocket 57', 'handshaking 58',
            'walking towards each other 59', 'walking apart from each other 60'
        ]
        
        self.action_names['coco'] = [
            '12~15', '15~18', '18~24', '24~30', '30up'
        ]

        self.font_sizes = {
            'ntu': 6,
            'coco': 8,
        }


    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) heatmap (hm)')
        logging.info('4) NTU skeleton (ns), 5) confusion matrix (cm), 6) action accuracy (ac)')
        logging.info('Please input the number (or name) of the function, q for quit: ')
        while True:
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 'hm' or cmd == 'heatmap':
                self.show_heatmap()
            elif cmd == '4' or cmd == 'ns' or cmd == 'NTU skeleton':
                self.show_NTU_skeleton()
            elif cmd == '5' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '6' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')


    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        if os.path.exists(data_file):
            self.all_data = np.load(data_file)
        else:
            self.all_data = None
            logging.info('')
            logging.error('Error: Do NOT exist this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        
        # 儲存完整資料以供heatmap儲存功能使用
        self.all_feature = self.all_data['feature']
        self.all_label = self.all_data['label']
        self.all_weight = self.all_data['weight']
        self.all_out = self.all_data['out']
        
        logging.info('*********************Video Name************************')
        logging.info(self.all_data['name'][self.args.visualization_sample])
        logging.info('')

        feature = self.all_data['feature'][self.args.visualization_sample,:,:,:,:]
        self.location = self.all_data['location']
        if len(self.location) > 0:
            self.location = self.location[self.args.visualization_sample,:,:,:,:]
        self.data = self.all_data['data'][self.args.visualization_sample,:,:,:,:]
        self.label = self.all_data['label']
        weight = self.all_data['weight']
        out = self.all_data['out']
        cm = self.all_data['cm']
        self.cm = cm.astype('int')  # 保持原始計數格式
        self.cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]  # 備用的正規化版本

        dataset = self.args.dataset.split('-')[0]
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class-1]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)   # CAM method
        self.result = self.result[self.args.visualization_class-1,:,:,:]


    def show_action_accuracy(self):
        # 計算準確率：對角線元素除以該行的總數
        accuracy = self.cm.diagonal() / self.cm.sum(axis=1)
        
        logging.info('Accuracy of each class:')
        logging.info('(Count format: correct_predictions / total_predictions)')
        for i in range(len(accuracy)):
            correct_count = self.cm.diagonal()[i]
            total_count = self.cm.sum(axis=1)[i]
            logging.info('{}: {:.4f} ({}/{})'.format(self.names[i], accuracy[i], correct_count, total_count))
        logging.info('')

        plt.figure()
        plt.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.ylabel('Accuracy')
        plt.title('Action Accuracy by Class')
        plt.show()


    def show_confusion_matrix(self):
        # 顯示原始計數的混淆矩陣（true 和 predict 對調版本）
        cm = self.cm.T  # 將混淆矩陣轉置，對調 true 和 predict
        show_name_x = self.names
        show_name_y = self.names

        plt.figure(figsize=(10, 8))
        font_size = self.font_size
        
        # 使用計數格式顯示混淆矩陣，增大數字字體
        sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', annot_kws={'fontsize':font_size+2},
                    cbar=True, square=True, linewidths=0.1, linecolor='black',
                    xticklabels=show_name_x, yticklabels=show_name_y)
        plt.xticks(fontsize=font_size, rotation=90)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Index of True Classes', fontsize=font_size)
        plt.ylabel('Index of Predict Classes', fontsize=font_size)
        plt.title('Confusion Matrix (Count Format) - True vs Predict Swapped', fontsize=font_size+2)
        plt.tight_layout()
        plt.show()
        
        # 同時顯示一些統計資訊
        logging.info('Confusion Matrix Statistics (True vs Predict Swapped):')
        logging.info('Total samples: {}'.format(cm.sum()))
        logging.info('Correct predictions: {}'.format(cm.diagonal().sum()))
        logging.info('Overall accuracy: {:.4f}'.format(cm.diagonal().sum() / cm.sum()))
        logging.info('')


    def show_NTU_skeleton(self):
        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([2,1,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12])
        result = np.maximum(self.result, 0)
        result = result/np.max(result)

        if len(self.args.visualization_frames) > 0:
            pause, frames = 10, self.args.visualization_frames
        else:
            pause, frames = 0.1, range(self.location.shape[1])

        plt.figure()
        plt.ion()
        for t in frames:
            if np.sum(self.location[:,t,:,:]) == 0:
                break

            plt.cla()
            plt.xlim(-50, 2000)
            plt.ylim(-50, 1100)
            plt.axis('off')
            plt.title('sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class:{}, actural_class:{}'.format(
                self.args.visualization_sample, self.names[self.args.visualization_class-1],
                t, self.probablity*100, self.pred_class, self.actural_class
            ))

            for m in range(M):
                x = self.location[0,t,:,m]
                y = 1080 - self.location[1,t,:,m]

                c = []
                for v in range(V):
                    r = result[t//4,v,m]
                    g = 0
                    b = 1 - r
                    c.append([r,g,b])
                    k = connecting_joint[v] - 1
                    plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=np.array([0.1,0.1,0.1]), linewidth=0.5, markersize=0)
                plt.scatter(x, y, marker='o', c=c, s=16)
            plt.pause(pause)
        plt.ioff()
        plt.show()


    def show_heatmap(self):
        # 檢查是否需要儲存所有validation樣本的heatmap
        if hasattr(self.args, 'visualization_heatmap_save') and self.args.visualization_heatmap_save:
            self.save_all_heatmaps()
            return
        
        # --- 數據準備 (與原程式碼相同) ---
        I, C, T, V, M = self.data.shape
        max_frame = T
        for t in range(T):
            if np.sum(self.data[:, :, t, :, :]) == 0:
                max_frame = t
                break

        num_persons = self.result.shape[-1]
        if num_persons == 0:
            print("沒有可以顯示的結果。")
            return

        fig, axes = plt.subplots(num_persons, 1, figsize=(10, 4 * num_persons), squeeze=False)

        # 顯示 pred_class 和 actural_class
        plt.suptitle(f'Predicted Class: {self.pred_class}, Actual Class: {self.actural_class}', fontsize=16)

        skeleton1 = self.result[:, :, 0]
        heat1 = np.zeros((max_frame // 4 * 4, V))
        for t in range(max_frame // 4):
            if t + 1 < skeleton1.shape[0]:
                d1 = (skeleton1[t + 1, :] - skeleton1[t, :]) / 4
            else:
                d1 = np.zeros_like(skeleton1[t, :])
            for i in range(4):
                heat1[t * 4 + i, :] = skeleton1[t, :] + d1 * i
                
        vmax = np.max(heat1) if np.max(heat1) > 0 else 1

        ax1 = axes[0, 0]
        im = ax1.imshow(heat1.T, cmap=plt.cm.plasma, vmin=0, vmax=vmax, aspect='auto')
        ax1.set_title('Person 1', fontsize=14)
        ax1.set_ylabel('Joints', fontsize=12)

        if num_persons > 1:
            ax1.tick_params(axis='x', labelbottom=False)

        if num_persons > 1:
            ax2 = axes[1, 0]
            skeleton2 = self.result[:, :, 1]
            heat2 = np.zeros((max_frame // 4 * 4, V))
            for t in range(max_frame // 4):
                if t + 1 < skeleton2.shape[0]:
                    d2 = (skeleton2[t + 1, :] - skeleton2[t, :]) / 4
                else:
                    d2 = np.zeros_like(skeleton2[t, :])
                for i in range(4):
                    heat2[t * 4 + i, :] = skeleton2[t, :] + d2 * i
            
            ax2.imshow(heat2.T, cmap=plt.cm.plasma, vmin=0, vmax=vmax, aspect='auto')
            ax2.set_title('Person 2', fontsize=14)
            ax2.set_ylabel('Joints', fontsize=12)

        axes[-1, 0].set_xlabel('Frames', fontsize=12)

        # fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01, shrink=0.8)

        plt.tight_layout()
        plt.show()

    def save_all_heatmaps(self):
        """儲存所有validation樣本的heatmap到指定資料夾"""
        save_dir = self.args.visualization_heatmap_save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f'Created directory: {save_dir}')

        num_samples = len(self.all_label)
        logging.info(f'Saving heatmaps for {num_samples} validation samples to {save_dir}')

        for sample_idx in range(num_samples):
            # 為每個樣本計算所有類別的結果
            feature = self.all_feature[sample_idx,:,:,:,:]
            true_class = self.all_label[sample_idx]
            pred_class = np.argmax(self.all_out[sample_idx])
            
            # 獲取樣本名稱
            sample_name = self.all_data['name'][sample_idx] if 'name' in self.all_data else f'sample_{sample_idx:04d}'
            # 清理檔案名稱中的非法字符
            safe_sample_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_sample_name = safe_sample_name.replace(' ', '_')
            
            # 使用真實類別來計算CAM
            result = np.einsum('kc,ctvm->ktvm', self.all_weight, feature)
            result_for_true_class = result[true_class,:,:,:]
            
            # 獲取樣本資料
            sample_data = self.all_data['data'][sample_idx,:,:,:,:]
            I, C, T, V, M = sample_data.shape
            
            # 計算有效幀數
            max_frame = T
            for t in range(T):
                if np.sum(sample_data[:, :, t, :, :]) == 0:
                    max_frame = t
                    break

            num_persons = result_for_true_class.shape[-1]
            if num_persons == 0:
                logging.warning(f'Sample {sample_idx}: No persons found, skipping')
                continue

            # 為每個人創建heatmap
            for person_idx in range(num_persons):
                skeleton = result_for_true_class[:, :, person_idx]
                heat = np.zeros((max_frame // 4 * 4, V))
                
                for t in range(max_frame // 4):
                    if t + 1 < skeleton.shape[0]:
                        d = (skeleton[t + 1, :] - skeleton[t, :]) / 4
                    else:
                        d = np.zeros_like(skeleton[t, :])
                    for i in range(4):
                        heat[t * 4 + i, :] = skeleton[t, :] + d * i

                # 創建並儲存圖片
                plt.figure(figsize=(12, 6))
                vmax = np.max(heat) if np.max(heat) > 0 else 1
                
                plt.imshow(heat.T, cmap=plt.cm.plasma, vmin=0, vmax=vmax, aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title(f'{safe_sample_name} Person {person_idx+1}\nTrue Class: {true_class+1}, Pred Class: {pred_class+1}', fontsize=14)
                plt.xlabel('Frames', fontsize=12)
                plt.ylabel('Joints', fontsize=12)
                
                # 儲存檔案 - 包含data name
                filename = f'{safe_sample_name}_true_{true_class+1}_pred_{pred_class+1}.png'
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()

            if (sample_idx + 1) % 10 == 0:
                logging.info(f'Processed {sample_idx + 1}/{num_samples} samples')

        logging.info(f'All heatmaps saved to {save_dir}')



    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample**********************')
        logging.info(wrong_sample)
        logging.info('')


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum)
        logging.info('')
        logging.info('Most important joints:')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:,:,1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum)
            logging.info('')
            logging.info('Most important joints:')
            logging.info(second_index)
            logging.info('')
