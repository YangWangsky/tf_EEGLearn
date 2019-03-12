Data for Visual WM experiment described in:
Bashivan, Pouya, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

FeatureMat_timeWin:	# 提取到的FFT power valuse
FFT power values extracted for three frequency bands (theta, alpha, beta). Features are arranged in band and electrodes order (theta_1, theta_2..., theta_64, alpha_1, alpha_2, ..., beta_64). There are seven time windows, features for each time window are aggregated sequentially (i.e. 0:191 --> time window 1, 192:383 --> time windw 2 and so on. Last column contains the class labels (load levels).
# 特征的排列顺序   (theta_(1~64),alphsa_(1~64),beta_(1~64)) * 7个时间窗口(flame)  即 64*3*7 = 1344 + 1(labels) = 1345


Neuroscan_locs_orig:
3 dimensional coordinates for electrodes on Neuroscan quik-cap.		# Neuroscan quik-cap上电极的三维坐标

trials_subNums:
contains subject numbers associated with each trial (used for leave-subject-out cross validation).	#每个trail对应的subject编号