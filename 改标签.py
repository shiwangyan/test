
with open("书签.txt", "r",encoding='utf-8') as file:
    lines = file.readlines()
    # print(lines)
    # file.close()
    modified_lines = []
    for line in lines:
        modified_line = ""
        i = 0
        pos = line.find("\\")
        if pos >= 0:
            # print(line[pos+1:len(line)])
            front = line.split("\\")[0]
            # print(front)
            origin_num = line[pos+1:len(line)]
            modified_num = int(origin_num) + 17
            modified_line = front + "\\" + str(modified_num) + "\n"
            # print(modified_line)
            modified_lines.append(modified_line)

        else:
            modified_lines.append(line)

# print(modified_lines)
with open("修改后书签.txt", "w",encoding='utf-8') as file:

    file.writelines(modified_lines)