# 根据混淆矩阵给出的四个指标，tp,fn,fp,tn
tp=0.98
fn=0.02
fp=0.01
tn=0.99
cat_precise=tp/(tp+fp)
cat_recall=tp/(tp+fn)
print("猫的精确率",cat_precise)
print("猫的召回率",cat_recall)