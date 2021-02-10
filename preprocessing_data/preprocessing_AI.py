import os
import fnmatch
import argparse
import json
import shutil
import time, datetime


def arg_parser():
    parser = argparse.ArgumentParser(description="insert directory path")
    parser.add_argument('-p', '--path', type=str, required=True, help='directory path')
    args = parser.parse_args()

    path = args.path

    return path

def create_folder():
    try:
        if not (os.path.isdir("error")):
            os.makedirs("error")
    except OSError:
        print("ERROR : CREATING directory. " + "error")

    try:
        if not (os.path.isdir("error/jpg_file_name_error")):
            os.makedirs("error/jpg_file_name_error")
    except OSError:
        print("ERROR : CREATING directory. " + "error/jpg_file_name_error")

    try:
        if not (os.path.isdir("error/json_file_name_error")):
            os.makedirs("error/jpg_file_name_error")
    except OSError:
        print("ERROR : CREATING directory. " + "error/jpg_file_name_error")

    try:
        if not (os.path.isdir("error/json_file_name_error")):
            os.makedirs("error/json_file_name_error")
    except OSError:
        print("ERROR : CREATING directory. " + "error/json_file_name_error")

    try:
        if not (os.path.isdir("error/json_exist_jpg_error")):
            os.makedirs("error/json_exist_jpg_error")
    except OSError:
        print("ERROR : CREATING directory. " + "error/json_exist_jpg_error")

    try:
        if not (os.path.isdir("error/jpg_exist_json_error")):
            os.makedirs("error/jpg_exist_json_error")
    except OSError:
        print("ERROR : CREATING directory. " + "error/jpg_exist_json_error")

    try:
        if not (os.path.isdir("error/mv_file")):
            os.makedirs("error/mv_file")
    except OSError:
        print("ERROR : CREATING directory. " + "error/mv_file")


def file_error_check(dir_path):
    print("================ preprocessing start ===============")
    jpg_all_count = 0
    json_all_count = 0

    jpg_json_exist_count = 0
    json_jpg_exist_count = 0

    jpg_exist_json_error_count = 0
    json_exist_jpg_error_count = 0

    jpg_file_name_collect_count = 0
    jpg_file_name_error_count = 0
    json_file_name_collect_count = 0
    json_file_name_error_count = 0

    jpg_0P_00_count = 0
    json_0P_00_count = 0

    json_new_version_count = 0
    json_old_version_count = 0


    create_folder()
    f_jpg_exist_json_error = open("error/f_jpg_exist_json_error.txt", 'w')
    f_jpg_json_collect = open("error/f_jpg_json_collect.txt", 'w')
    f_jpg_file_name_error = open("error/f_jpg_file_name_error.txt", 'w')
    f_json_jpg_collect = open("error/f_json_jpg_collect.txt", 'w')
    f_json_exist_jpg_error = open("error/f_json_exist_jpg_error.txt", 'w')
    f_json_file_name_error = open("error/f_json_file_name_error.txt", 'w')
    f_all_count = open("error/f_all_count.txt", 'w')
    f_except_list = open("error/f_except_list.txt", 'w')
    f_json_old_version_list = open("error/f_json_old_version_list.txt", 'w')

    move_file_list = []
    error_list = []

    json_error_count = 0
    progress = 1

    # file name error check
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            print("file error check : ", filename, " -- num : all ==", str(progress), " : ",len(files))
            file_name, exp = os.path.splitext(filename)

            if exp.lower() == ".jpg":
                # print(root + "\\" + filename)
                jpg_all_count += 1

                # file name error check, file exist check, error file move to error folder
                if fnmatch.fnmatch(file_name, "DC[0-9][0-9][0-9][0-9]_2[0-9]2[0-1]-[0-1][0-9]-"
                                              "[0-3][0-9] [0-9][0-9][0-9][0-9][0-9][0-9]_[0,A-Z][0,A-Z]"):
                    jpg_file_name_collect_count += 1
                    if os.path.exists(root + "\\" + file_name + ".json"):
                        f_jpg_json_collect.write(root + "\\" + file_name + exp + "\n")
                        jpg_json_exist_count += 1
                    else:
                        f_jpg_exist_json_error.write(root + "\\" + file_name + exp + "\n")
                        jpg_exist_json_error_count += 1
                        move_file_list.append("error/jpg_exist_json_error/" + file_name + exp)
                else:
                    jpg_file_name_error_count += 1
                    f_jpg_file_name_error.write(root + "\\" + file_name + exp + "\n")
                    move_file_list.append("error/jpg_file_name_error/" + file_name + exp)

                 # 00 0P file check
                if filename.split(".")[0][-1] == "P":
                    if os.path.exists(root + "\\" + filename.split(".")[0][:-1] + "0.jpg"):
                        jpg_0P_00_count += 1
                        move_file_list.append("error/mv_file/" + filename.split(".")[0][:-1] + "0.jpg")

            elif exp.lower() == ".json":
                json_all_count += 1

                # file name error check, file exist check, error file move to error folder
                if fnmatch.fnmatch(file_name, "DC[0-9][0-9][0-9][0-9]_2[0-9]2[0-1]-[0-1][0-9]-"
                                              "[0-3][0-9] [0-9][0-9][0-9][0-9][0-9][0-9]_[0,A-Z][0,A-Z]"):
                    json_file_name_collect_count += 1
                    if os.path.exists(root + "\\" + file_name + ".jpg") or \
                            os.path.exists(root + "\\" + file_name + ".JPG"):
                        f_json_jpg_collect.write(root + "\\" + file_name + exp + "\n")
                        json_jpg_exist_count += 1
                    else:
                        f_json_exist_jpg_error.write(root + "\\" + file_name + exp + "\n")
                        json_exist_jpg_error_count += 1
                        move_file_list.append("error/json_exist_jpg_error/" + file_name + exp)
                else:
                    json_file_name_error_count += 1
                    f_json_file_name_error.write(root + "\\" + file_name + exp + "\n")
                    move_file_list.append("error/json_file_name_error/" + file_name + exp)

                # if exist  0P and 00 together remove 00 file
                if filename.split(".")[0][-1] == "P":
                    if os.path.exists(root + "\\" + filename.split(".")[0][:-1] + "0.json"):
                        json_0P_00_count += 1
                        move_file_list.append("error/mv_file/" + filename.split(".")[0][:-1] + "0.json")

                try:
                    json_file = open(root + "\\" + filename, 'rt', encoding="utf-8")
                    json_data = json.loads(json_file.read())
                    json_file.close()

                    json_write_file = open(root + "\\" + filename, 'w', encoding="utf-8")

                    if json_data.get("annotations"):
                        json_new_version_count += 1
                        annotations = json_data.get("annotations")

                        for i in range(len(annotations)):
                            annotation = annotations[i]
                            # segmentation error check
                            segmentations = annotation["segmentation"]

                            # if segmentation points have minus value, modify segmentation points
                            for i2 in range(len(segmentations)):
                                segmentation = segmentations[i2]
                                if segmentation[0] <= 0:
                                    segmentation[0] = 0
                                if segmentation[1] <= 0:
                                    segmentation[1] = 0
                    else:
                        json_old_version_count += 1
                        f_json_old_version_list.write(root + "\\" + filename + "\n")

                        shapes_all = json_data.get("shapes")
                        for i in range(len(shapes_all)):
                            shapes = shapes_all[i]
                            points = shapes["points"]
                            for i2 in range(len(points)):
                                point = points[i2]
                                if point[0] <= 0:
                                    point[0] = 0
                                if point[1] <= 0:
                                    point[1] = 0
                        json.dump(json_data, json_write_file, ensure_ascii=False, indent=2)
                        json_write_file.close()
                # if json file format error, modify json file format
                except:
                    try:
                        f = open(root + "\\" + filename, 'r', encoding="UTF-8")
                        lines = f.readlines()
                        f.close()
                        except_category_list = []
                        except_i = 0
                        except_i2 = 0
                        except_i3 = 0
                        except_case_1 = False
                        for i in range(len(lines)):
                            if lines[i].rstrip() == '  "category": [':
                                except_category_list.append(lines[i])
                                except_i += i
                                if len(except_category_list) > 1:
                                    except_i2 += i
                            if lines[i][0].rstrip() == "}":
                                if not lines[i].rstrip() == "}":
                                    except_case_1 = True
                                    except_i3 += i

                        if len(except_category_list) > 1:
                            f5 = open(root + "\\" + filename, 'w', encoding="utf-8")
                            for i in range(0, except_i - except_i2):
                                f5.write(str(lines[i]).rstrip() + "\n")
                            for i2 in range(except_i2, len(lines)):
                                f5.write(str(lines[i2]).rstrip() + "\n")
                            f5.close()

                        if except_case_1:
                            f6 = open(root + "\\" + filename, 'w', encoding="utf-8")
                            for i in range(0, except_i3 + 1):
                                if i == except_i3:
                                    lines[i] = "}"
                                    f6.write(str(lines[i]).rstrip())
                                else:
                                    f6.write(str(lines[i]).rstrip() + "\n")
                            f6.close()

                        if lines[0].rstrip() == "{" and lines[-1].rstrip() == "}":
                            if lines[-2].rstrip() == "  ]":
                                if not lines[-3].rstrip() == "    }":
                                    f2 = open(root + "\\" + filename, 'w', encoding="utf-8")
                                    for i in range(len(lines) - 2):
                                        f2.write(str(lines[i]).rstrip() + "\n")
                                    f2.close()
                        elif lines[-1].rstrip() == "}}":
                            lines[-1] = "}"
                            f3 = open(root + "\\" + filename, 'w', encoding="utf-8")
                            for i in range(len(lines)):
                                f3.write(str(lines[i]).rstrip() + "\n")
                            f3.close()
                        elif lines[-1].rstrip() == "}  }":
                            lines[-1] = "}"
                            f4 = open(root + "\\" + filename, 'w', encoding="utf-8")
                            for i in range(len(lines)):
                                f4.write(str(lines[i]).rstrip() + "\n")
                            f4.close()
                        else:
                            f_except_list.write(root + "\\" + filename + "\n")

                        json_file = open(root + "\\" + filename, "rt", encoding="utf-8")
                        json_data = json.loads(json_file.read())
                        json_file.close()
                        json_write_file = open(root + "\\" + filename, 'w', encoding="utf-8")

                        if json_data.get("annotations"):
                            json_new_version_count += 1
                            annotations = json_data.get("annotations")
                            for i in range(len(annotations)):
                                annotation = annotations[i]

                                segmentations = annotation["segmentation"]

                                for i2 in range(len(segmentations)):
                                    segmentation = segmentations[i2]
                                    if segmentation[0] <= 0:
                                        segmentation[0] = 0
                                    if segmentation[1] <= 0:
                                        segmentation[1] = 0
                        else:
                            json_old_version_count += 1
                            f_json_old_version_list.write(root + "\\" + filename + "\n")

                            shapes_all = json_data.get("shapes")
                            for i in range(len(shapes_all)):
                                shapes = shapes_all[i]
                                points = shapes["points"]
                                for i2 in range(len(points)):
                                    point = points[i2]
                                    if point[0] <= 0:
                                        point[0] = 0
                                    if point[1] <= 0:
                                        point[1] = 0

                            json.dump(json_data, json_write_file, ensure_ascii=False, indent=2)
                            json_write_file.close()
                    # Create as a file if it cannot be fixed
                    except:
                        error_list.append(root + "\\" + filename)
                        json_error_count += 1
            progress += 1


    f_all_count.write("jpg_all_count : " + str(jpg_all_count) + "\n"
                      + "json_all_count : " + str(json_all_count) + "\n"
                      + "jpg_json_exist_count : " + str(jpg_json_exist_count) + "\n"
                      + "json_jpg_exist_count : " + str(json_jpg_exist_count) + "\n"
                      + "jpg_exist_json_error_count : " + str(jpg_exist_json_error_count) + "\n"
                      + "json_exist_jpg_error_count : " + str(json_exist_jpg_error_count) + "\n"
                      + "jpg_file_name_collect_count : " + str(jpg_file_name_collect_count) + "\n"
                      + "jpg_file_name_error_count : " + str(jpg_file_name_error_count) + "\n"
                      + "json_file_name_collect_count : " + str(json_file_name_collect_count) + "\n"
                      + "json_file_name_error_count : " + str(json_file_name_error_count) + "\n"
                      + "jpg_0P_00_count : " + str(jpg_0P_00_count) + "\n"
                      + "json_0P_00_count : " + str(json_0P_00_count) + "\n"
                      + "json_new_version_count : " + str(json_new_version_count) + "\n"
                      + "json_old_version_count : " + str(json_old_version_count) + "\n")

    f_jpg_exist_json_error.close()
    f_jpg_json_collect.close()
    f_jpg_file_name_error.close()
    f_json_jpg_collect.close()
    f_json_exist_jpg_error.close()
    f_json_file_name_error.close()
    f_all_count.close()
    f_except_list.close()

    print("================ preprocessing Done ===============")
    print(json_error_count)

    return move_file_list, error_list

def move_file(dir_path, move_file_list):
    print("================ file move start ===============")

    # f_move_file_list = open("error/f_move_file_list.txt", 'w')
    process_num = 1
    for i in range(len(move_file_list)):
        # f_move_file_list.write()
        print("move file path: ", move_file_list[i], process_num ,":", len(move_file_list))
        file_name = move_file_list[i].split("/")[-1].rstrip()
        if os.path.exists(dir_path + "\\" + file_name):
            shutil.move(dir_path + "\\" + file_name, move_file_list[i].rstrip())
        process_num += 1



def remove_file(dir_path, remove_file_list):
    print("================ json format error file move start ===============")
    process_num = 1

    try:
        if not (os.path.isdir("error/remove")):
            os.makedirs("error/remove")
    except OSError:
        print("ERROR : CREATING directory. " + "error/remove")

    for i in range(len(remove_file_list)):
        print("remove file path : ", remove_file_list[i], process_num, ": ", len(remove_file_list))
        file_name = remove_file_list[i].split("\\")[-1].rstrip()
        filename = file_name.split(".")[0]
        print(filename)
        if os.path.exists(dir_path + "\\" + file_name):
            shutil.move(remove_file_list[i].rstrip(), "error/remove/" + file_name)
            if os.path.exists(dir_path + "\\" + filename + ".jpg"):
                shutil.move(dir_path + "\\" + filename + ".jpg", "error/remove/" + filename + ".jpg")


def json_old_version_check(dir_path):
    process = 1
    f = open("error/f_json_old_version_list2.txt", 'w')
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            file_name, exp = os.path.splitext(filename)

            if exp.lower() == ".json":
                print(filename, process, ":", len(files))
                json_file = open(root + "\\" + filename, 'r', encoding="utf-8")
                json_data = json.loads(json_file.read())
                json_file.close()

                if not json_data.get("annotations"):
                    f.write(root + "\\" + filename + "\n")
            process += 1
    f.close()

def json_old_version_move(dir_path, json_move_file_list):
    print("================ file move start ===============")

    # f_move_file_list = open("error/f_move_file_list.txt", 'w')
    process_num = 1
    for i in range(len(json_move_file_list)):
        # f_move_file_list.write()
        print("move file path: ", json_move_file_list[i], process_num ,":", len(json_move_file_list))
        file_name = json_move_file_list[i].split("\\")[-1].rstrip()
        filename, exp = file_name.split(".")
        if os.path.exists(dir_path + "\\" + file_name):
            shutil.move(dir_path + "\\" + file_name, "error/mv_file/" + filename + ".json")
            if os.path.exists(dir_path + "\\" + filename + ".jpg"):
                shutil.move(dir_path + "\\" + filename + ".jpg", "error/mv_file/" + filename + ".jpg")
        process_num += 1

    print("================ file move Done ===============")


def read_json_to_yolo_txt(dir_path):
    f_exclude_class_id = open("../exclude_class_id.txt", 'r')

    class_dict = {}
    class_id = ""

    while True:
        line = f_exclude_class_id.readline()
        if not line: break
        line = line.rstrip("\n")
        list_a = line.split(",")
        class_dict[list_a[1]] = list_a[0]

    f_exclude_class_id.close()
    file_count = 0
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            file_name, exp = os.path.splitext(filename)
            if exp.lower() == ".json":
                file_count += 1
                print("filename : ",filename, " file_count : ", file_count, " / ", len(files))
                json_file = open(root + "/" + filename, "rt", encoding="UTF-8")
                json_data = json.loads(json_file.read())

                category = json_data["category"]
                annotations = json_data["annotations"]
                images = json_data["images"]
                images_height = images["height"]
                images_width = images["width"]

                f = open(root + "/" + file_name + ".txt", 'w')

                for i in range(len(category)):
                    category_all = category[i]
                    annotations_all = annotations[i]
                    category_id = category_all['id']

                    if category_id in class_dict:
                        segmentation = annotations_all['segmentation']
                        poly_or_rect = annotations_all['id']
                        list_b = poly_or_rect.split("_")
                        shape = list_b[0]

                        for i2 in range(len(class_dict)):
                            class_id = class_dict[category_id]
                        list_x = []
                        list_y = []

                        if shape == "rectangle":
                            x_y_min = segmentation[0]
                            x_y_max = segmentation[1]

                            x_min = x_y_min[0]
                            y_min = x_y_min[1]
                            x_max = x_y_max[0]
                            y_max = x_y_max[1]

                            box = (x_min, x_max, y_min, y_max)
                            convert_box = convert((images_width, images_height), box)

                        elif shape == "polygon":
                            for i3 in range(len(segmentation)):
                                seg_ = segmentation[i3]
                                x = seg_[0]
                                y = seg_[1]
                                list_x.append(x)
                                list_y.append(y)

                            x_min = min(list_x)
                            x_max = max(list_x)
                            y_min = min(list_y)
                            y_max = max(list_y)

                            box = (x_min, x_max, y_min, y_max)
                            convert_box = convert((images_width, images_height), box)

                        if i == len(category) - 1:
                            f.write(class_id + " " + " ".join([str(a) for a in convert_box]))
                        else:
                            f.write(class_id + " " + " ".join([str(a) for a in convert_box]) + "\n")
                f.close()
                json_file.close()

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    if x <= 0:
        x = 0
    if y <= 0:
        y = 0
    if x >= 1:
        x = 1
    if y >= 1:
        y = 1

    if w <= 0:
        w = 0
    if h <= 0:
        h = 0
    if w >= 1:
        w = 1
    if h >= 1:
        h = 1

    return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))


if __name__ == '__main__':
    dir_path = arg_parser()

    start_time = time.time()
    move_file_list, error_list = file_error_check(dir_path)
    f = open("error/move_file_list.txt", 'w')
    for i in range(len(move_file_list)):
        f.write(move_file_list[i] + "\n")
    f.close()

    f2 = open("error/error_list.txt", 'w')
    for i in range(len(error_list)):
        f2.write(error_list[i] + "\n")
    f2.close()

    f3 = open("error/move_file_list.txt", 'r')
    move_file_list2 = f3.readlines()
    f3.close()
    move_file(dir_path, move_file_list2)

    f4 = open("error/error_list.txt", 'r')
    remove_file_list = f4.readlines()
    f4.close()
    remove_file(dir_path, remove_file_list)

    json_old_version_check(dir_path)

    f5 = open("error/f_json_old_version_list2.txt", 'r')
    json_move_file_list = f5.readlines()
    f5.close()
    json_old_version_move(dir_path, json_move_file_list)

    read_json_to_yolo_txt(dir_path)

    end_time = time.time()

    sec = end_time - start_time
    times = str(datetime.timedelta(seconds=sec))
    print(times)
