#-*- coding: utf-8 -*-
import json
import os
import openpyxl

sub_category_dict = {
"flatness_A":0, "flatness_B":0, "flatness_C":0, "flatness_D":0, "flatness_E":0,
"walkway_paved":0, "walkway_block":0, "paved_state_broken":0, "paved_state_normal":0,
"block_state_broken":0, "block_state_normal":0, "block_kind_bad":0, "block_kind_good":0,
"outcurb_rectangle":0, "outcurb_slide":0, "outcurb_rectangle_broken":0, "outcurb_slide_broken":0,
"restspace":0, "sidegap_in" :0, "sidegap_out" :0, "sewer_cross" :0, "sewer_line" :0,
"brailleblock_dot":0, "brailleblock_line":0, "brailleblock_dot_broken":0, "brailleblock_line_broken" :0,
"continuity_tree":0, "continuity_manhole":0, "ramp_yes":0, "ramp_no":0,
"bicycleroad_broken":0, "bicycleroad_normal":0, "planecrosswalk_broken":0, "planecrosswalk_normal" :0,
"steepramp":0, "bump_slow":0, "bump_zigzag":0, "weed":0, "floor_normal":0, "floor_broken":0,
"flowerbed":0, "parkspace":0, "tierbump":0, "stone":0, "enterrail":0, "fireshutter":0,

"stair_normal":0, "stair_broken":0, "wall":0, "window_sliding":0, "window_casement":0,
"pillar":0, "lift":0, "door_normal":0, "door_rotation":0, "lift_door":0, "resting_place_roof":0,
"reception_desk":0, "protect_wall_protective":0, "protect_wall_guardrail":0, "protect_wall_kickplate":0,
"handle_vertical":0, "handle_lever":0, "handle_circular":0,
"lift_button_normal":0, "lift_button_openarea":0, "lift_button_layer":0, "lift_button_emergency":0,
"direction_sign_left":0, "direction_sign_right":0, "direction_sign_straight":0, "direction_sign_exit":0,
"sign_disabled_toilet":0, "sign_disabled_parking":0, "sign_disabled_elevator":0,
"sign_disabled_ramp":0, "sign_disabled_callbell":0, "sign_disabled_icon":0,
"braille_sign":0, "chair_multi":0, "chair_one":0, "chair_circular":0, "chair_back":0, "chair_handle":0,
"number_ticket_machine":0, "beverage_vending_machine":0, "beverage_desk":0, "trash_can":0, "mailbox":0
}
sub_category_to_main_category_dict = {
"flatness":["flatness_A","flatness_B","flatness_C","flatness_D","flatness_E"], "walkway":["walkway_paved","walkway_block"],
"paved_state":["paved_state_broken","paved_state_normal"], "block_state":["block_state_broken","block_state_normal"],
"block_kind":["block_kind_bad","block_kind_good"], "outcurb":["outcurb_rectangle","outcurb_slide","outcurb_rectangle_broken","outcurb_slide_broken"],
"restspace":["restspace"], "sidegap":["sidegap_in","sidegap_out"], "sewer":["sewer_cross","sewer_line"],
"brailleblock":["brailleblock_dot","brailleblock_line","brailleblock_dot_broken","brailleblock_line_broken"],
"continuity":["continuity_tree","continuity_manhole"], "ramp":["ramp_yes","ramp_no"], "bicycleroad":["bicycleroad_broken","bicycleroad_normal"],
"planecrosswalk":["planecrosswalk_broken","planecrosswalk_normal"], "steepramp":["steepramp"], "bump":["bump_slow","bump_zigzag"],
"weed":["weed"], "floor":["floor_normal","floor_broken"], "flowerbed":["flowerbed"], "parkspace":["parkspace"],
"tierbump":["tierbump"], "stone":["stone"], "enterrail":["enterrail"], "fireshutter":["fireshutter"],

"stair":["stair_normal"], "stair_broken":["stair_broken"], "wall":["wall"], "window":["window_sliding","window_casement"],
"pillar":["pillar"], "lift":["lift"], "door":["door_normal","door_rotation"], "lift_door":["lift_door"],
"resting_place_roof":["resting_place_roof"], "reception_desk":["reception_desk"],
"protect_wall":["protect_wall_protective","protect_wall_guardrail","protect_wall_kickplate"],
"handle":["handle_vertical","handle_lever","handle_circular"],
"lift_button":["lift_button_normal","lift_button_openarea","lift_button_layer","lift_button_emergency"],
"direction_sign":["direction_sign_left","direction_sign_right","direction_sign_straight","direction_sign_exit"],
"sign_disabled":["sign_disabled_toilet","sign_disabled_parking","sign_disabled_elevator","sign_disabled_ramp","sign_disabled_callbell","sign_disabled_icon"],
"braille_sign":["braille_sign"], "chair":["chair_multi","chair_one","chair_circular"], "chair_back":["chair_back"],
"chair_handle":["chair_handle"], "number_ticket_machine":["number_ticket_machine"], "beverage_vending_machine":["beverage_vending_machine"],
"beverage_desk":["beverage_desk"], "trash_can":["trash_can"], "mailbox":["mailbox"]
}
main_category_dict = {
"flatness":0, "walkway":0, "paved_state":0, "block_state":0, "block_kind":0, "outcurb":0, "restspace":0,
"sidegap":0, "sewer":0, "brailleblock":0, "continuity":0, "ramp":0, "bicycleroad":0, "planecrosswalk":0,
"steepramp":0, "bump":0, "weed":0, "floor":0, "flowerbed":0, "parkspace":0, "tierbump":0,
"stone":0, "enterrail":0, "fireshutter":0,

"stair":0, "stair_broken":0, "wall":0, "window":0, "pillar":0, "lift":0, "door":0, "lift_door":0,
"resting_place_roof":0, "reception_desk":0, "protect_wall":0, "handle":0, "lift_button":0,
"direction_sign":0, "sign_disabled":0, "braille_sign":0, "chair":0, "chair_back":0, "chair_handle":0,
"number_ticket_machine":0, "beverage_vending_machine":0, "beverage_desk":0, "trash_can":0, "mailbox":0
}
filtering_folder = [""]

def category_counting(path):
    jpg_count = 0
    json_count = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [dir for dir in dirs if dir.lower() not in filtering_folder]
        for file_name in files:
            if file_name.split(".")[0][-1] == "s":
                continue

            ext = os.path.splitext(file_name)[1]
            if ext.lower() == ".json":
                print(file_name)
                json_file = open(root + "\\" + file_name, "rt", encoding="UTF8")
                jsonString = json.load(json_file)

                if jsonString.get("annotations") != None:
                    category_id = jsonString.get("annotations")
                    category_key = "category_id"
                else:
                    category_id = jsonString.get("shapes")
                    category_key = "label"

                for shapes in category_id:
                    sub_category_dict[shapes[category_key]] += 1
                    for key, value in sub_category_to_main_category_dict.items():
                        if shapes[category_key] in value:
                            main_category_dict[key] += 1
                json_count += 1
            elif ext.lower() == ".jpg":
                print(file_name)
                jpg_count += 1

    write_excel(sub_category_dict,json_count,jpg_count)
    print(sub_category_dict)
    print(main_category_dict)

def write_excel(sub_category_dict, json_count, jpg_count):
    one_folder_wb = openpyxl.load_workbook('statistics_form.xlsx')
    one_folder_total_sheet = one_folder_wb['statistics']
    idx = 5

    for _, value in sub_category_dict.items():
        if idx == 51:
            idx = 54
        one_folder_total_sheet['G' + str(idx)] = int(value)
        idx += 1
    one_folder_total_sheet['D98'] = jpg_count
    one_folder_total_sheet['G98'] = json_count

    if not (os.path.isdir("./output")):
        os.makedirs(os.path.join("./output"))
    one_folder_wb.save('./output/statistics.xlsx')

if __name__ == "__main__":
    path = input("Path : ")
    print("Input Path : " + path)
    category_counting(path)
