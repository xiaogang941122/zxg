# 删除无用的列，增加条件判断避免KeyError
if 'time' in df.columns:
    df.drop(columns="time", inplace=True)
else:
    print("列 'time' 不存在，跳过删除操作。")

# 继续后续的代码逻辑...