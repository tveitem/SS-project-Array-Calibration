
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import easygui
import seaborn as sns


# Function which reads an mcc file of measurements from VeriSoft taken with original calibration file and extracts the relevant data:

# In[2]:


def read_file():
    file_name = easygui.fileopenbox()
    with open(file_name) as mcc_file:
        mcc_reader = mcc_file.readlines()
    my_dict = {} #Empty dictionary
    data_switch = 0 #"switch" telling Python to read/not read lines
    for line in mcc_reader:
        if 'SCAN_OFFAXIS_INPLANE' in line:
            offset = float(line.split('=')[1])
            my_dict[str(offset)] = {} #Creating a dictionary of the y-values / off-axis offsets
        if 'BEGIN_DATA' in line:
            data_switch = 1
            continue   
        if 'END_DATA' in line:
            data_switch = 0
        if data_switch == 1:
            x = line.split()[0]
            dose = line.split()[1]
            my_dict[str(offset)][str(x)] = float(dose) #New sub-dictionary for x-values in which each is assigned to a dose
    readings = np.zeros((53,53)) #Extracting y-values (offsets)
    for i,y_pos in enumerate(list(my_dict.keys())):
        x_list = list(my_dict[y_pos].keys())
        for j,x_pos in enumerate(x_list):
            dose_value = my_dict[y_pos][x_pos]
            if (i % 2) == 0:
                readings[i][j*2] = dose_value    
            else:
                readings[i][(j*2)+1] = dose_value     
    return readings


# Selection of measurement files:

# In[3]:


centre_1 = read_file()
centre_2 = read_file()
centre_3 = read_file()
centre_4 = read_file()

quadrant1_1 = read_file()
quadrant1_2 = read_file()
quadrant1_3 = read_file()

quadrant2_1 = read_file()
quadrant2_2 = read_file()
quadrant2_3 = read_file()

quadrant3_1 = read_file()
quadrant3_2 = read_file()
quadrant3_3 = read_file()

quadrant4_1 = read_file()
quadrant4_2 = read_file()
quadrant4_3 = read_file()


# Selection of an mcc file of the required calibration file format as a template:

# In[4]:


template_file = easygui.fileopenbox()


# Calculation of the mean of measurements at each position:

# In[5]:


centre = np.mean(np.array([centre_1, centre_2, centre_3, centre_4]), axis = 0)
plt.imshow(centre)
plt.title('Centre mean reading')
plt.show()

quadrant1 = np.mean(np.array([quadrant1_1, quadrant1_2, quadrant1_3]), axis = 0)
plt.imshow(quadrant1)
plt.title('Q1 mean reading')
plt.show()

quadrant2 = np.mean(np.array([quadrant2_1, quadrant2_2, quadrant2_3]), axis = 0)
plt.imshow(quadrant2)
plt.title('Q2 mean reading')
plt.show()

quadrant3 = np.mean(np.array([quadrant3_1, quadrant3_2, quadrant3_3]), axis = 0)
plt.imshow(quadrant3)
plt.title('Q3 mean reading')
plt.show()

quadrant4 = np.mean(np.array([quadrant4_1, quadrant4_2, quadrant4_3]), axis = 0)
plt.imshow(quadrant4)
plt.title('Q4 mean reading')
plt.show()


# General parameters: (Specific to the PTW OCTAVIUS 1500 array)

# In[26]:


row_of_centre_cell = 26
column_of_centre_cell = 26
total_no_of_cells_per_row = 53
middle_row_index = 26


# Function which rotates a reading by 45 degrees:

# In[25]:


def rotate(array):
    all_neg = []
    all_pos = []
    for i in range(len(array)):
        diag_neg = np.diag(array,-i) #Extracting diagonals to the left of centre
        diag_pos = np.diag(array, i) #Extracting diagonals to the right of centre
        all_neg.append(diag_neg)
        all_pos.append(diag_pos)
    all_neg = all_neg[::-1] #Reversing order of diagonals to the left of centre
    all_neg = all_neg[:-1]  #Removing the last one in the list, i.e. the central one, so it is not included twice
    all_diag = all_neg + all_pos #Gathering all diagonals into one list
    diagonals = all_diag[::2] #Extracting the diagonals with 1's:
    diagonals = diagonals[::-1]
    diamond = np.zeros(array.shape) #Placing into new array as horizontal rows
    for j in range(len(diagonals)):
        len_row_zeros = len(diamond[j])
        len_of_diag = len(diagonals[j])
        zeros_keep = len_row_zeros - len_of_diag
        zeros_keep_half = zeros_keep / 2
        if zeros_keep_half == 0:
            diamond[j] = diagonals[j]
        else:
            diamond[j, int(zeros_keep_half):-(int(zeros_keep_half))] = diagonals[j]
    return diamond


# Calibration function:

# In[39]:


def calibration(OCQ1, OCQ2, diamond_init, diamond1_init, diamond2_init, calibrating = 'initial', rotation = 'no'): #OCQ1 and OCQ2 are output flucuation correction factors
    diamond = diamond_init
    diamond1 = diamond1_init*OCQ1
    diamond2 = diamond2_init*OCQ2
    diamond[diamond == 0] = 1 #Avoiding division by 0
    diamond1[diamond1 == 0] = 1 
    diamond2[diamond2 == 0] = 1 
    
    #HORIZONTAL CALIBRATION
    CF_array_Q1 = np.zeros(diamond.shape) #An array of zeros into which values calculated in the loop will be placed
    CF_array_Q1[:,column_of_centre_cell] = 1 #Setting the centre column to 1
    for i in range(len(diamond[row_of_centre_cell,(column_of_centre_cell+1):-1])):
        CF_r = (CF_array_Q1[:,column_of_centre_cell+i]) * (diamond[:, (column_of_centre_cell+1)+i] / diamond1[:, (column_of_centre_cell+1)+i])
        CF_array_Q1[:, (column_of_centre_cell+1)+i] = CF_r   #Placing the corrected values into the array CF_array_Q1 on its right-hand side
        CF_l = (CF_array_Q1[:,column_of_centre_cell-i]) * (diamond1[:, column_of_centre_cell-i] / diamond[:, (column_of_centre_cell)-i]) #(init. value is CF_array_Q1[:,column_of_centre_cell] first since i=0)
        CF_array_Q1[:, (column_of_centre_cell-1)-i] = CF_l   #Placing the corrected values into the array CF_array_Q1 on its left-hand side

    #VERTICAL CALIBRATION
    CF_array_Q2 = np.zeros(diamond.shape) #An array of zeros into which CFs calculated in the loop will be placed
    CF_array_Q2[row_of_centre_cell,:] = 1 #Setting the entire centre row to 1
    for i in range(len(diamond[(row_of_centre_cell+1):-1,column_of_centre_cell])): 
        CF_up = (CF_array_Q2[row_of_centre_cell+i,:]) * (diamond2[row_of_centre_cell+i,:] / diamond[row_of_centre_cell+i,:]) 
        CF_array_Q2[(row_of_centre_cell+1)+i,:] = CF_up #Placing the corrected valuesinto the array CF_array_Q1 on its bottom side
        CF_do = (CF_array_Q2[row_of_centre_cell-i,:]) * (diamond[(row_of_centre_cell-1)-i,:] / diamond2[(row_of_centre_cell-1)-i,:])
        CF_array_Q2[(row_of_centre_cell-1)-i,:] = CF_do #Placing the corrected valuesinto the array CF_array_Q1 on its top side
    
    #COMBINING CALIBRATIONS
    blank = np.ones((53,53))
    Q1_central_row_array = blank * CF_array_Q1[row_of_centre_cell,:]
    CF_array_final_Q1Q2 = CF_array_Q2 * Q1_central_row_array
    
    #MASK FOR EXTRACTING RElEVANT CELLS FROM CFs IN ROTATED FORM:
    identity = np.eye(2,2)
    mask = np.tile(identity, (27,27))[:-1,:-1] #Tiling the array to the required size
    mask_all_neg = []
    mask_all_pos = []
    for i in range(len(mask)):
        mask_diag_neg = np.diag(mask,-i) #Extracting diagonals to the left of centre
        mask_diag_pos = np.diag(mask, i) #Extracting diagonals to the right of centre
        mask_all_neg.append(mask_diag_neg)
        mask_all_pos.append(mask_diag_pos)
    mask_all_neg = mask_all_neg[::-1] #Reversing order of diagonals to the left of centre
    mask_all_neg = mask_all_neg[:-1]  #Removing the last one in the list, i.e. the central one, so it is not included twice
    mask_all_diag = mask_all_neg + mask_all_pos #Gathering all diagonals into one list
    mask_diagonals = mask_all_diag[::2] #Extracting the diagonals with 1's
    mask_diamond = np.zeros(mask.shape) #Placing into a new array as horizontal rows
    for j in range(len(mask_diagonals)):
        mask_len_row_zeros = len(mask_diamond)
        mask_len_of_diag = len(mask_diagonals[j])
        mask_zeros_keep = mask_len_row_zeros - mask_len_of_diag
        mask_zeros_keep_half = mask_zeros_keep / 2
        if mask_zeros_keep_half == 0:
            mask_diamond[j] = mask_diagonals[j]
        else:
            mask_diamond[j, int(mask_zeros_keep_half):-(int(mask_zeros_keep_half))] = mask_diagonals[j]
    
    #USING MASK TO EXTRACT THE TRUE CFs FROM ROTATED FORM AND PUTTING INTO CHEQUER BOARD STRUCTURE:
    all_diags_length_27 = []
    for i in range(-26,27): #takes 26 on either side of central diagonal (0)
        mask_diag = np.diag(mask_diamond, k=i) #extracting the diagonals from the mask
        CF_diag = np.diag(CF_array_final_Q1Q2, k=i) #extracting the diagonals from the calibration diamond
        indices = np.where(mask_diag == 1) #is a list of 2 entries where the first entry [0] is an array of the indices of interest for each diagonal
        CF_values = CF_diag[indices[0]] #For each diagonal in CF_array I am choosing the cells/indices that are the same as the indices for which the mask is 1, i.e. the cells which should go back onto chequer board
        all_diags_length_27.append(CF_values) #Appending the diagonals of the right respective lengths to this list
    #Putting these CFs into chequer board structure
    calib_true_2D_dose_dist = np.zeros((53,53)) #empty array
    for i,diag in enumerate(all_diags_length_27):
        if len(diag)==27:
            calib_true_2D_dose_dist[::2,i]=diag
        elif len(diag)==26:
            calib_true_2D_dose_dist[1::2,i]=diag
    
    #QUANTIFYING DIFFERENCES BETWEEN RECONSTRUCTED PROFILES AND REFERENCE PROFILES:
    if calibrating == 'initial':
        wt_charge_LR_unnorm = np.load('LR_reference.npy') #'LR_reference.npy' is a reference left-right profile
        wt_pos_LR = np.load('LR_x.npy')              #'LR_x.npy' is the x-values of the reference left-right profile
        wt_charge_GT_unnorm = np.load('TG_reference.npy') #'TG_reference.npy' is a reference target-gun profile, y-values
        wt_pos_GT = np.load('TG_x.npy')              #'TG_x.npy' is the x-values of the reference target-gun profile
        array_x = np.linspace(130,-130, 27) #Subset of xarray of water tank
        wt_charge_GT = wt_charge_GT_unnorm/wt_charge_GT_unnorm[31] #Normalizing
        wt_charge_LR = wt_charge_LR_unnorm/wt_charge_LR_unnorm[31]
        
        if rotation == 'yes':
            calib_true_2D_dose_dist = np.rot90(calib_true_2D_dose_dist,2)

        LR_calib_true_dose_profile = calib_true_2D_dose_dist[26,::2] #Left-right central profile from the reconstructed dose distribution
        GT_calib_true_dose_profile = calib_true_2D_dose_dist[::2,26] #Target-gun central profile from the reconstructed dose distribution      
       
        LR_wt_profile_array_resolution = np.interp(array_x, wt_pos_LR, wt_charge_LR) #Reference profile charge values at the x-values of the reconstruted readings
        GT_wt_profile_array_resolution = np.interp(array_x, wt_pos_GT, wt_charge_GT) 

        LR_diff = LR_calib_true_dose_profile[5:22] - LR_wt_profile_array_resolution[5:22] #Differences
        GT_diff = GT_calib_true_dose_profile[5:22] - GT_wt_profile_array_resolution[5:22] #Selecting 80% of the profile (of clinical interest), avoiding penumbra region 
       
        std_LR_Q1Q2 = np.std(LR_diff) #Standard deviation
        std_GT_Q1Q2 = np.std(GT_diff)

        return std_LR_Q1Q2, std_GT_Q1Q2
    
    else:
        
        return calib_true_2D_dose_dist


# Implementation of calibration:

# In[42]:


def total_calibration(centre_ave_reading, Q1_ave_reading, Q2_ave_reading, Q3_ave_reading, Q4_ave_reading):
    centre_rot = np.rot90(centre_ave_reading,2) #Rotating arrays for the calibration which uses Q3 and Q4 measurements
    quadrant3_rot = np.rot90(Q3_ave_reading,2)
    quadrant4_rot = np.rot90(Q4_ave_reading,2)
    
    diamond = rotate(centre_ave_reading) #Rotating the array measurements
    diamond_rot = rotate(centre_rot)
    diamond1_initial = rotate(Q1_ave_reading)
    diamond2_initial = rotate(Q2_ave_reading)
    diamond3_initial_rot = rotate(quadrant3_rot)
    diamond4_initial_rot = rotate(quadrant4_rot)
    
    plt.figure(figsize = (17,6)) #Plotting the diamonds
    plt.subplot(1,5,1)
    plt.imshow(diamond)
    plt.axvline(x=26)
    plt.axhline(y=26)
    plt.title('Centre')
    plt.subplot(1,5,2)
    plt.imshow(diamond1_initial)
    plt.axvline(x=26)
    plt.axhline(y=26)
    plt.title('Quadrant 1')
    plt.subplot(1,5,3)
    plt.imshow(diamond2_initial)
    plt.axvline(x=26)
    plt.axhline(y=26)
    plt.title('Quadrant 2')
    plt.figure(figsize = (15,5))
    plt.subplot(1,5,4)
    plt.imshow(diamond3_initial_rot)
    plt.axvline(x=26)
    plt.axhline(y=26)
    plt.title('Quadrant 3')
    plt.subplot(1,5,5)
    plt.imshow(diamond4_initial_rot)
    plt.axvline(x=26)
    plt.axhline(y=26)
    plt.title('Quadrant 4')
    plt.show()
    
    #LOOPING THROUGH DIFFERENT COMBINATIONS OF OUTPUT CORRECTION FACTORS
    output_corrections = np.arange(0.9800,1.0205,0.0005)
    LR_std_list_Q1Q2 = []
    GT_std_list_Q1Q2 = []
    std_average_list_Q1Q2 = []
    LR_std_list_Q3Q4 = []
    GT_std_list_Q3Q4 = []
    std_average_list_Q3Q4 = []
    for i in output_corrections:
        for j in output_corrections:
            LRstd_Q1Q2, GTstd_Q1Q2 = calibration(i, j, diamond, diamond1_initial, diamond2_initial, 'initial', 'no') #Normalized true profile
            LRstd_Q3Q4, GTstd_Q3Q4 = calibration(i, j, diamond_rot, diamond3_initial_rot, diamond4_initial_rot, 'initial', 'yes') 
            #Q1 and Q2:
            LR_std_list_Q1Q2.append(LRstd_Q1Q2)
            GT_std_list_Q1Q2.append(GTstd_Q1Q2)
            std_average_Q1Q2 = (LRstd_Q1Q2 + GTstd_Q1Q2) / 2
            std_average_list_Q1Q2.append((i,j,std_average_Q1Q2))
            #Q3 and Q4:
            LR_std_list_Q3Q4.append(LRstd_Q3Q4)
            GT_std_list_Q3Q4.append(GTstd_Q3Q4)
            std_average_Q3Q4 = (LRstd_Q3Q4 + GTstd_Q3Q4) / 2
            std_average_list_Q3Q4.append((i,j,std_average_Q3Q4))
    std_averages_only_Q1Q2 = [i[2] for i in std_average_list_Q1Q2]
    std_averages_only_Q3Q4 = [i[2] for i in std_average_list_Q3Q4]     
    
    #OPTIMIZING
    #Q1 & Q2:
    min_ave_std_Q1Q2 = np.min(std_averages_only_Q1Q2) #Finding the optimized combination of output correction factors by finding the smallest average standard deviations
    index_of_min_Q1Q2 = np.where(std_averages_only_Q1Q2 == min_ave_std_Q1Q2)[0][0]
    OC_combination_min_Q1Q2 = std_average_list_Q1Q2[index_of_min_Q1Q2][0:2]
    output_correction_Q1 = OC_combination_min_Q1Q2[0]
    print('Output Correction Factor for Q1 Measurements: ',output_correction_Q1)
    output_correction_Q2 = OC_combination_min_Q1Q2[1]
    print('Output Correction Factor for Q2 Measurements: ',output_correction_Q2)
    print()
    #Q3 & Q4:
    min_ave_std_Q3Q4 = np.min(std_averages_only_Q3Q4)
    index_of_min_Q3Q4 = np.where(std_averages_only_Q3Q4 == min_ave_std_Q3Q4)[0][0]
    OC_combination_min_Q3Q4 = std_average_list_Q3Q4[index_of_min_Q3Q4][0:2]
    output_correction_Q3 = OC_combination_min_Q3Q4[0]
    print('Output Correction Factor for Q3 Measurements: ',output_correction_Q3)
    output_correction_Q4 = OC_combination_min_Q3Q4[1]
    print('Output Correction Factor for Q4 Measurements: ',output_correction_Q4)

    #REPEATING CALIBRATION USING THE OPTIMIZED CORRECTION FACTORS
    calib_true_2D_dose_dist_Q1Q2 = calibration(output_correction_Q1, output_correction_Q2, diamond, diamond1_initial, diamond2_initial, 'final', 'no')
    calib_true_2D_dose_dist_Q3Q4_rot = calibration(output_correction_Q3, output_correction_Q4, diamond_rot, diamond3_initial_rot, diamond4_initial_rot, 'final', 'no')
    calib_true_2D_dose_dist_Q3Q4 = np.rot90(calib_true_2D_dose_dist_Q3Q4_rot,2) #Rotating corrected dose distribution obtained with measurements from Q3 & Q4 back to true detector positions
    
    #AVERAGING THE TWO CORRECTED DOSE DISTRIBUTIONS
    calib_true_2D_dose_dist = (calib_true_2D_dose_dist_Q1Q2 + calib_true_2D_dose_dist_Q3Q4) / 2
    
    plt.figure(figsize = (15,4)) #Plotting the reconstructed two central profiles
    plt.subplot(1,2,1)
    plt.plot(calib_true_2D_dose_dist[26,::2])
    plt.title('Reconstructed Left-Right Profile \n', fontsize=16)
    plt.ylabel('dose (normalized)', fontsize=13)
    plt.xlabel('detector position', fontsize=13)
    plt.subplot(1,2,2)        
    plt.plot(calib_true_2D_dose_dist[::2,26])
    plt.ylabel('dose (normalized)', fontsize=13)
    plt.xlabel('detector position', fontsize=13)
    plt.title('Reconstructed Target-Gun Profile \n', fontsize=16)
    plt.show()
    
    return calib_true_2D_dose_dist    


# In[43]:


final_calib_true_2D_dose_dist = total_calibration(centre, quadrant1, quadrant2, quadrant3, quadrant4)


# Replacement of any zeros and negative values outside the radiation field so as to be acceptable to PTW-ArrayCal:

# In[21]:


final_calib_true_2D_dose_dist[np.where(final_calib_true_2D_dose_dist == 0)] = final_calib_true_2D_dose_dist[0,2]
final_calib_true_2D_dose_dist[np.where(final_calib_true_2D_dose_dist < 0)] = final_calib_true_2D_dose_dist[0,2]


# Removal of detector-less gaps for compatibility with required mcc file format:

# In[ ]:


list_of_values_wo_zeros = []
for k,row in enumerate(final_calib_true_2D_dose_dist):
    if k%2 == 0: #Row is even
        values = row[::2]
        coords = np.linspace(-130,130,27)
    else:
        values = row[1::2]
        coords = np.linspace(-125,125,26)
    list_of_values_wo_zeros.append(values)


# Writing of an mcc file by inserting corrected dose values:

# In[14]:


scanning = 0
data_switch = 0
row_number = -1
with open(r'\Users\Home\Desktop\NAME.mcc', 'w') as file: #File path and name
    with open(template_file) as my_file:
        lines = my_file.readlines()
        new_lines = []
        for line in lines:
            if 'BEGIN_SCAN' in line:
                new_lines.append(line)
                scanning = 1
                continue
            elif 'END_SCAN' in line:
                new_lines.append(line)
                scanning = 0
                continue

            if scanning == 1:
                if 'BEGIN_DATA' in line:
                    new_lines.append(line)
                    data_switch = 1
                    row_number += 1
                    col_number = 0
                    continue
                elif 'END_DATA' in line:
                    new_lines.append(line)
                    data_switch = 0
                    continue

                if data_switch == 1:
                    split_line = line.split('\t')
                    split_line[5] = np.str.upper(str((np.format_float_scientific(list_of_values_wo_zeros[row_number][col_number],4))))                
                    line = '\t'.join(split_line)
                    col_number += 1
            new_lines.append(line)
    file.writelines(new_lines)        


# Comparson to reference profiles for evaluation of calibration:

# In[60]:


array_x = np.linspace(130, -130, 27) #Subset of x-array of reference
wt_charge_GT = (wt_charge_GT_unnorm/wt_charge_GT_unnorm[31]) #Normalizing
wt_charge_LR = wt_charge_LR_unnorm/wt_charge_LR_unnorm[31]
norm_centre = centre / centre[26,26]
LR_calib_true_dose_profile = final_calib_true_2D_dose_dist[26,::2]
GT_calib_true_dose_profile = final_calib_true_2D_dose_dist[::2,26]
LR_centre = norm_centre[26,::2]
GT_centre = norm_centre[::2,26]
plt.figure(figsize = (20,7))

plt.subplot(1,2,1)
plt.plot(wt_pos_LR, wt_charge_LR, label='Reference')
plt.plot(array_x, LR_calib_true_dose_profile, 'r.', label='Calibrated Array')
plt.ylabel('charge / dose (normalized)', fontsize=19)
plt.xlabel('position from CAX [mm]', fontsize=19)
plt.title('Left-Right \n', fontsize = 25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(loc='right', fontsize=19)
plt.subplot(1,2,2)
plt.plot(-wt_pos_GT, wt_charge_GT, label='Reference')
plt.plot(-array_x, GT_calib_true_dose_profile, 'r.', label='Calibrated Array')
plt.xlabel('position from CAX [mm]', fontsize=19)
plt.ylabel('charge / dose (normalized)', fontsize=19)
plt.title('Target-Gun \n', fontsize=25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(loc='right', fontsize=19)
plt.show()

LR_wt_profile_array_resolution = np.interp(array_x, wt_pos_LR, wt_charge_LR)
GT_wt_profile_array_resolution = np.interp(array_x, wt_pos_GT, wt_charge_GT)

LR_differences = abs(LR_wt_profile_array_resolution[5:22] - LR_calib_true_dose_profile[5:22])
GT_differences = abs(GT_wt_profile_array_resolution[5:22] - GT_calib_true_dose_profile[5:22])

percentile95_lr = np.percentile(LR_differences, 95)
print('95th percentile central left-right profile: ',percentile95_lr)
percentile95_gt = np.percentile(GT_differences, 95)
print('95th percentile central target-gun profile: ',percentile95_gt)

std_LR = np.std(LR_differences)
std_GT = np.std(GT_differences)
print()
print('Standard deviation central left-right profile: ',std_LR)
print('Standard deviation central target-gun profile: ',std_GT)

plt.style.use('seaborn')
sns.kdeplot(LR_differences, label='Left-right')
sns.kdeplot(GT_differences, label='Target-gun')
plt.xlabel('relative dose difference', fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('density', fontsize=15)
plt.legend(fontsize=15)
plt.title('Discrepancies reference values and corrected \n array values along central profiles', fontsize=15)
plt.show()


# Comparson to profiles obtained with original calibration file for evaluation of calibration:

# In[54]:


plt.figure(figsize = (20,7))
plt.subplot(1,2,1)
plt.plot(LR_calib_true_dose_profile, 'r', label='Calibrated')
plt.plot(LR_centre,'bo', label='Uncalibrated')
plt.xlabel('detector number',fontsize=19)
plt.ylabel('dose (normalized)',fontsize=19)
plt.title('Left-Right \n', fontsize=25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=19)
plt.subplot(1,2,2)    
plt.plot(GT_calib_true_dose_profile, 'r', label='Calibrated')
plt.plot(GT_centre,'bo', label='Uncalibrated')
plt.xlabel('detector number',fontsize=19)
plt.ylabel('dose (nomalized)',fontsize=19)
plt.title('Target-Gun \n', fontsize=25)
plt.legend(fontsize=19)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

