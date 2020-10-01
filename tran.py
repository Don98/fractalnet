def deal_name():
    with open("name.txt","r") as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    nums = [1,2,4,8]
    pos  = [0,2,3,3,2,3,3,1,2,3,3,1,2,3,3]
    with open("name.txt","a") as f:
        for i in pos:
            f.write("conv" + str(i) + "_" + str(nums[i])+"\n")
            nums[i] += 1
        for i in pos:
            f.write("conv" + str(i) + "_" + str(nums[i])+"\n")
            nums[i] += 1
        for i in pos:
            f.write("conv" + str(i) + "_" + str(nums[i])+"\n")
            nums[i] += 1
    
deal_name()
# exit()
def get_name():
    with open("name.txt","r") as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    return data

name = get_name()
result = ""

for j in range(4):
    for i in range(j*15,(j+1)*15):
    # for i in name:
        result += "\nstate_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".conv1.weight'] = torch.from_numpy(name_weights['" + name[i] + "']['weight'])\n"
        result += "\nstate_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".conv1.bias'] = torch.from_numpy(name_weights['" + name[i] + "']['bias'])\n"

        result += "state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.running_var'] = from_numpy(name_weights['batch_" + name[i] + "']['running_var'])\n"
        result += "state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.running_mean'] = from_numpy(name_weights['batch_" + name[i] + "']['running_mean'])\n"
        result += "state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.weight'] = torch.ones_like(state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.running_var'])\n"
        result += "state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.bias'] = torch.ones_like(state_dict['the_block" + str(j+1) + ".0." + name[i-j*15] + ".bn1.running_var'])\n"

with open("load_dic.py","a") as f:
    f.write(result)