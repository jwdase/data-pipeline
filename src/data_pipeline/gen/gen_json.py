"""
Generate  all the .json to run this code

scene_makeup: 
{
    "static" : ["dining_room_table"]
    "observed" : []
    "hidden" : []
    "target" : str
}

scene_priors :
{
    "object" : {
        "position" : [
            [x, y, z, qx, qz, qw],
            [x, y, z, qx, qz, qw]
        ]
        "pos_std" : 0.1
        "rot_std" : 0.1
        "x_max" : 0.5, "x_min" : -0.5,
        "y_max" : 0.5, "y_min" : -0.5,
    }
    }
}


--> Position is taken from other occluders position and then placed at correct y
--> Most occluded object is usually the target
"""


import json
import math

# Threshold Value
THRESHOLD = 0.1

def create_prior_makeup(root, scene):
    scene_comp_loc = root / scene / "results" / "target.json"
    scene_truth_loc = root / scene / "data" / f"{scene}_truth.json"

    with open(scene_comp_loc, "r") as f:
        scene_comp = json.load(f)

    with open(scene_truth_loc, "r") as f:
        truths = json.load(f)

    make_prior_and_makeup(root, scene, scene_comp, truths)
    

def make_prior_and_makeup(root, scene, scene_comp, truths):
    """
    Creates scene_makeup.json and scene_priors.json according to 
    specification that enables it to work with scene physics library

    Note using this function requires interfacing with it. You must declare
    what object is the occluded target

    Args:
        root : [str] Experiment numer of form "exp05"
        scene : [str] Scene name
        scene_comp : [dict] contains occlusion level of objects in scene
        truths : [dict] contains the true locations of each object
    """
    


    lowest_occlusion = math.inf
    lowest_occlusion_obj = None

    options = []

    for scene_obj in scene_comp["objects"]:
        if scene_obj["visible_fraction"] < lowest_occlusion:
            lowest_occlusion_obj = scene_obj["name"]
            lowest_occlusion = scene_obj["visible_fraction"]
        if scene_obj["visible_fraction"] < THRESHOLD:
            options.append((scene_obj["name"], scene_obj["visible_fraction"]))

    target = get_occluded_target(options, lowest_occlusion_obj)


    scene_makeup = {
        "static" : ["dining_room_table"],
        "observed" : [
            obj["name"]
            for obj in scene_comp["objects"]
            if obj["name"] != target
        ],
        "hidden" : [target],
        "target" : target
    }

    # Correct Z Height
    y = truths[target][2]


    scene_priors ={
        target : {
            "position" : [
                [truths[name][0], truths[name][1], y, truths[name][3], truths[name][4], truths[name][5], truths[name][6]]
                for name in scene_makeup["observed"]
            ],
            "pos_std" : 0.1,
            "rot_std" : 0.1,
            "x_max" : 0.5, "x_min" : -0.5,
            "y_max" : 0.5, "y_min" : -0.5,
        }
    }

    save_files(root, scene, scene_makeup, scene_priors)

    return None

def save_files(root, scene, scene_makeup, scene_priors):
    """
    Just save these two json files
    """

    start = root / scene / "data" / scene

    with open(f"{start}_makeup.json", "w", encoding="utf-8") as f:
        json.dump(scene_makeup, f, indent=4)

    with open(f"{start}_priors.json", "w", encoding="utf-8") as f:
        json.dump(scene_priors, f, indent=4)

    return None

# Choice if >= 2
def get_occluded_target(options, lowest):
    """
    Figures out what the occluded object is for a given scene

    Args:
        options : [list] of potential (name, frac)
        lowest : [str] name of lowest

    Return:
        name : [str] lowest_occluded
    """
    if len(options) >= 2:
        choices = {
        i : scene_stats
        for i, scene_stats in enumerate(options)
    }

        for i, data in choices.items():
            print(f"Option: {i} is object {data[0]} with score {data[1]}")


        print("Please enter best choice")
        answer = None

        while answer is None:
            try:
                response = int(input("Number: "))
            except ValueError:
                response = None
                pass

            if response in choices.keys():
                try:
                    response_2 = int(input("Verify: "))
                except ValueError:
                    response_2 = None
            else:
                response_2 = None

            if (response == 0 or response) and response == response_2:
                answer = response
            else:
                print("Please enter the correct option")

        return choices[answer][0]

    else:
       return lowest
    
if __name__ == "__main__":
    pass