import json
import pandas as pd

with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_Baseline_Adam_BCE_epochs=100_lr_0.001_aug_0.json', 'r') as data:
    data1 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_Baseline_Adam_BCE_epochs=100_lr_0.0001_aug_0.json', 'r') as data:
    data2 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_Baseline_Adam_BCE_epochs=100_lr_0.001_aug_1.json', 'r') as data:
    data3 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_Baseline_Adam_BCE_epochs=100_lr_0.0001_aug_1.json', 'r') as data:
    data4 = json.load(data)

with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_BCE_epochs=100_lr_0.001_aug_0.json', 'r') as data:
    data5 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_BCE_epochs=100_lr_0.0001_aug_0.json', 'r') as data:
    data6 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_BCE_epochs=100_lr_0.001_aug_1.json', 'r') as data:
    data7 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_BCE_epochs=100_lr_0.0001_aug_1.json', 'r') as data:
    data8 = json.load(data)

with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_Baseline_Adam_FocalLoss_epochs=100_lr_0.001_aug_0.json', 'r') as data:
    data9 = json.load(data)

with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_FocalLoss_epochs=100_lr_0.001_aug_0.json', 'r') as data:
    data11 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_FocalLoss_epochs=100_lr_0.0001_aug_0.json', 'r') as data:  
    data10 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_Adam_FocalLoss_epochs=100_lr_0.001_aug_1.json', 'r') as data:
    data12 = json.load(data)

with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_RMSprop_BCE_epochs=100_lr_0.001_aug_0.json', 'r') as data:
    data15 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_RMSprop_BCE_epochs=100_lr_0.0001_aug_0.json', 'r') as data:
    data16 = json.load(data)
with open('/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_UNet_RMSprop_BCE_epochs=100_lr_0.001_aug_1.json', 'r') as data:
    data17 = json.load(data)


df = pd.DataFrame(columns=['Model', 'Optimizer', 'Loss fn', 'Augmentation', 'LR', 'loss', 'dice', 'IoU', 'accuracy', 'sensivity', 'specificity'])
### augmentation ### ### learning rate ###
df.loc[1] = ['Baseline', 'Adam', 'BCE', 'No', '0.001', max(data1['loss_test']), max(data1['dice_test']), max(data1['IoU_test']), max(data1['accuracy_test']), max(data1['sensivity_test']), max(data1['specificity_test'])]
df.loc[2] = ['Baseline', 'Adam', 'BCE', 'No', '0.0001', max(data2['loss_test']), max(data2['dice_test']), max(data2['IoU_test']), max(data2['accuracy_test']), max(data2['sensivity_test']), max(data2['specificity_test'])]
df.loc[3] = ['Baseline', 'Adam', 'BCE', 'Yes', '0.001', max(data3['loss_test']), max(data3['dice_test']), max(data3['IoU_test']), max(data3['accuracy_test']), max(data3['sensivity_test']), max(data3['specificity_test'])]
df.loc[4] = ['Baseline', 'Adam', 'BCE', 'Yes', '0.0001', max(data4['loss_test']), max(data4['dice_test']), max(data4['IoU_test']), max(data4['accuracy_test']), max(data4['sensivity_test']), max(data4['specificity_test'])]

df.loc[5] = ['UNet', 'Adam', 'BCE', 'No', '0.001', max(data5['loss_test']), max(data5['dice_test']), max(data5['IoU_test']), max(data5['accuracy_test']), max(data5['sensivity_test']), max(data5['specificity_test'])]
df.loc[6] = ['UNet', 'Adam', 'BCE', 'No', '0.0001', max(data6['loss_test']), max(data6['dice_test']), max(data6['IoU_test']), max(data6['accuracy_test']), max(data6['sensivity_test']), max(data6['specificity_test'])]
df.loc[7] = ['UNet', 'Adam', 'BCE', 'Yes', '0.001', max(data7['loss_test']), max(data7['dice_test']), max(data7['IoU_test']), max(data7['accuracy_test']), max(data7['sensivity_test']), max(data7['specificity_test'])]
df.loc[8] = ['UNet', 'Adam', 'BCE', 'Yes', '0.0001', max(data8['loss_test']), max(data8['dice_test']), max(data8['IoU_test']), max(data8['accuracy_test']), max(data8['sensivity_test']), max(data8['specificity_test'])]
##################
### loss function ###
df.loc[9]  = ['Baseline', 'Adam', 'Focal Loss', 'No', '0.001', max(data9['loss_test']), max(data9['dice_test']), max(data9['IoU_test']), max(data9['accuracy_test']), max(data9['sensivity_test']), max(data9['specificity_test'])]

df.loc[10] = ['UNet', 'Adam', 'Focal Loss', 'No', '0.001', max(data11['loss_test']), max(data11['dice_test']), max(data11['IoU_test']), max(data11['accuracy_test']), max(data11['sensivity_test']), max(data11['specificity_test'])]
df.loc[11] = ['UNet', 'Adam', 'Focal Loss', 'No', '0.0001', max(data10['loss_test']), max(data10['dice_test']), max(data10['IoU_test']), max(data10['accuracy_test']), max(data10['sensivity_test']), max(data10['specificity_test'])]
df.loc[12] = ['UNet', 'Adam', 'Focal Loss', 'Yes', '0.001', max(data12['loss_test']), max(data12['dice_test']), max(data12['IoU_test']), max(data12['accuracy_test']), max(data12['sensivity_test']), max(data12['specificity_test'])]
##################
### optimizer ###
df.loc[13] = ['UNet', 'RMSprop', 'BCE', 'No', '0.001', max(data15['loss_test']), max(data15['dice_test']), max(data15['IoU_test']), max(data15['accuracy_test']), max(data15['sensivity_test']), max(data15['specificity_test'])]
df.loc[14] = ['UNet', 'RMSprop', 'BCE', 'No', '0.0001', max(data16['loss_test']), max(data16['dice_test']), max(data16['IoU_test']), max(data16['accuracy_test']), max(data16['sensivity_test']), max(data16['specificity_test'])]
df.loc[15] = ['UNet', 'RMSprop', 'BCE', 'Yes', '0.001', max(data17['loss_test']), max(data17['dice_test']), max(data17['IoU_test']), max(data17['accuracy_test']), max(data17['sensivity_test']), max(data17['specificity_test'])]
##################

df = df.round({'loss': 4, 'dice': 4, 'IoU': 4, 'accuracy': 4, 'sensivity': 4, 'specificity': 4})
df.to_csv('/zhome/6d/e/184043/DLCV/2/project/results/ablation_DRIVE.csv', index=False)
print(df)
