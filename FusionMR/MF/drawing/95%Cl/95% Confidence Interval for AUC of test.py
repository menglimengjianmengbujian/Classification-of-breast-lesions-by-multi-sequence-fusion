import  math

def confidence_level(auc_value ,positive_samples ,negative_samples):
    # 置信水平
    confidence_level = 0.95
    Q1 = auc_value / (2 - auc_value)
    Q2 = (2 * auc_value ** 2) / (1 + auc_value)
    # 计算标准误差
    standard_error = math.sqrt((auc_value * (1 - auc_value) +
                                (positive_samples - 1) * (Q1 - auc_value ** 2) +
                                (negative_samples - 1) * (Q2 - auc_value ** 2))
                               / (positive_samples * negative_samples))

    # 根据置信水平计算Z分数
    z_score = 1.96  # 对于95%置信水平

    # 计算百分之九十五的置信区间
    lower_bound = auc_value - z_score * standard_error
    upper_bound = auc_value + z_score * standard_error

    print(f"95% Confidence Interval for AUC: {lower_bound:.4f} - {upper_bound:.4f}")

print('  95% Confidence Interval for AUC of test model')
#融合模型
T1_T2_E_auc_score= 0.8595061022120518
T1_T2_E_label=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
print(f'三个融合模型：{T1_T2_E_auc_score:.4f}')
confidence_level(T1_T2_E_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

# 都是T1
T1_auc_score= 0.7865497076023392
print(f'都是T1：{T1_auc_score:.4f}')
confidence_level(T1_auc_score,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

# 都是T2
T2_auc_score= 0.35988749046529367
print(f'都是T2：{T2_auc_score:.4f}')
confidence_level(T2_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

# 都是enhance
E_auc_score= 0.6763205568268498
print(f'都是enhance：{E_auc_score :.4f}')
confidence_level(E_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

# T1+T2
T1_T2_auc_score= 0.8057621408593949
print(f'T1+T2：{T1_T2_auc_score:.4f}')
confidence_level(T1_T2_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

#T1+enhance
T1_E_auc_score= 0.8356375540300025
print(f'T1+enhance：{T1_E_auc_score:.4f}')
confidence_level(T1_E_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))

# T2和enhance融合
T2_E_auc_score= 0.5761346300533944
print(f'T2和enhance融合：{T2_E_auc_score:.4f}')
confidence_level(T2_E_auc_score ,T1_T2_E_label.count(0) ,T1_T2_E_label.count(1))



