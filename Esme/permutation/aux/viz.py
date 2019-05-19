def plot_test():
    import matplotlib.pyplot as plt
    ax = plt.subplot()
    ax.plot([1,2,3,4])
    plt.close()
    print('Test plt successfully')

def landscape_summary(landscape_data):
    assert type(landscape_data) == list
    n = len(landscape_data)
    for i in range(n):
        print('Beta is: ', landscape_data[i]['beta'], 'Training: ', landscape_data[i]['others'][0]['train_acc'],
              'Test: ', landscape_data[i]['others'][0]['test_acc'], 'Baseline PD vector: ', landscape_data[i]['pd_vector_data'])
    # dump_data(graph, dataset, dataname, beta=-1, still_dump='yes', skip='no')
def landscape_summary(landscape_data):
    assert type(landscape_data) == list
    n = len(landscape_data)
    for i in range(n):
        print('Beta is: ', landscape_data[i]['beta'], 'Training: ', landscape_data[i]['others'][0]['train_acc'],
              'Test: ', landscape_data[i]['others'][0]['test_acc'], 'Baseline PD vector: ', landscape_data[i]['pd_vector_data'])
    # dump_data(graph, dataset, dataname, beta=-1, still_dump='yes', skip='no')

def get_meshdata(start=87, end=177):
    f = open('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/landscape.txt')
    lines = f.readlines()
    lines = lines[start: end]
    lis = []
    for line in lines:
        new_line = line.replace('array', 'np.array')
        lis += [eval(new_line)]
    x = lis
    meshdata = {}
    for i in range(len(x)):
        assert x[i][0][2] == 0
        assert x[i][0][4] == 0
        x_ = x[i][0][1]
        y_ = x[i][0][3]
        z_ = x[i][0][0]
        meshdata[(x_, y_, z_)] = x[i][1]
    # print(type(meshdata), len(meshdata))
    return meshdata
# meshdata = get_meshdata(start=195, end=285)
def viz_meshdata_(graph):
    # not the mesh viz
    import matplotlib.pyplot as plt
    xy_ = np.array(meshdata.keys())
    x = xy_[:,0]
    y = xy_[:,1]
    z = xy_[:,2]
    t = meshdata.values()
    for i in range(len(x)):
        assert meshdata[x[i], y[i], z[i]] == t[i]
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x, y, z, s=10, c=t)
    fig.colorbar(im)
    ax.set_xlabel('ricci'); ax.set_ylabel('cc'); ax.set_zlabel('deg')
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/'
    make_direct(direct)
    filename = 'viz_land.png'
    plt.savefig(direct+filename)
    plt.close()
# viz_meshdata('nci1')

# Viz graph
def peekgraph(g, key='fv'):
    # input is nx.graph, output is the node and its node value
    for v, data in sorted(g.nodes(data=True), key=lambda x: x[1][key], reverse=True):
        print (v, data)
# peekgraph(graphs[77][1])
def draw_graph(g):
    # input is nx.graph
    import matplotlib.pyplot as plt
    import networkx as nx
    nx.draw(g)
    plt.draw()
    plt.show()
# draw_graph(graphs[77][1])
def viz_kernel(k):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # Make an array with ones in the shape of an 'X'
    a = k

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # Bilinear interpolation - this will look blurry
    ax1.imshow(a, cmap=cm.Greys_r)

    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(a, interpolation='nearest', cmap=cm.Greys_r)

    plt.show()

# MDS related
def color_map(i):
    if i == 1:
        return 0.1
    if i == 0:
        return 'b'
    if i == 2:
        return 0.6
    if i == 3:
        return 0.9
    elif i == -1:
        return 'r'
def new_label(graph):
    if graph == 'mutag':
        label = [0, 2, 3, 4, 5, 7, 8, 10,  13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 29, 33, 34, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47, 55, 56, 58, 60, 64, 65, 68, 71, 72, 76, 78, 79, 81, 82, 83, 85, 87, 88, 91, 93, 94, 95, 99, 100,  103, 104, 105, 107, 108, 111, 112, 113, 114, 116, 117, 118, 119, 120, 124, 150, 155, 162, 163, 164, 165, 167, 169, 170, 186 #np.array([0.    , 0.3879, 0.    , 0.6121, 0.    ] achieves almost 100 accuracy
] # [0, 0.5, 0, 0.5, 0] x>0
    if graph == 'protein_data':
         # label = [1, 2, 3, 4, 8, 9, 10, 11, 12, 14, 15, 16, 19, 21, 23, 24, 28, 29, 30, 31, 32, 33, 35, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 50, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 67, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109, 110, 112, 113, 114, 116, 117, 118, 121, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 140, 141, 144, 145, 146, 147, 148, 150, 151, 152, 155, 158, 163, 164, 165, 166, 168, 169, 170, 172, 173, 175, 176, 178, 181, 182, 184, 186, 187, 189, 191, 192, 195, 196, 202, 203, 204, 205, 207, 209, 211, 214, 215, 216, 217, 218, 219, 220, 221, 222, 225, 226, 228, 229, 230, 236, 238, 240, 241, 242, 243, 247, 249, 250, 251, 253, 255, 256, 257, 259, 262, 264, 265, 266, 271, 276, 278, 279, 280, 282, 286, 287, 288, 289, 290, 291, 293, 297, 298, 299, 301, 303, 310, 311, 312, 313, 315, 316, 317, 318, 320, 322, 323, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 338, 341, 342, 343, 344, 345, 346, 349, 350, 352, 354, 357, 358, 359, 360, 363, 364, 365, 366, 367, 368, 369, 372, 373, 375, 378, 379, 380, 382, 384, 385, 388, 389, 390, 391, 393, 394, 395, 396, 400, 401, 402, 403, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 423, 424, 425, 426, 429, 430, 433, 434, 436, 439, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 461, 462, 464, 465, 468, 471, 472, 473, 474, 477, 478, 479, 482, 485, 486, 490, 494, 495, 496, 497, 498, 500, 501, 504, 505, 506, 507, 511, 512, 513, 514, 515, 516, 517, 518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 545, 546, 547, 548, 549, 550, 552, 553, 554, 556, 557, 558, 559, 560, 561, 562, 563, 565, 568, 571, 572, 573, 575, 576, 578, 580, 582, 583, 586, 589, 590, 591, 594, 596, 597, 598, 600, 601, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 619, 620, 622, 623, 624, 625, 627, 628, 629, 632, 633, 635, 636, 637, 638, 641, 645, 646, 647, 649, 650, 652, 654, 657, 658, 660, 661, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681, 682, 683, 684, 685, 687, 688, 689, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 704, 705, 706, 707, 709, 710, 711, 712, 713, 715, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 735, 736, 737, 739, 740, 741, 742, 743, 744, 745, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 759, 762, 763, 764, 765, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 778, 779, 780, 781, 783, 784, 785, 786, 787, 788, 789, 791, 792, 793, 794, 795, 796, 797, 798, 799, 801, 803, 804, 805, 806, 807, 808, 809, 810, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 829, 830, 831, 832, 833, 834, 836, 837, 839, 840, 841, 842, 843, 845, 846, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 932, 933, 934, 935, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 956, 957, 958, 959, 960, 961, 962, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 986, 988, 989, 991, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1018, 1019, 1020, 1021, 1022, 1023, 1025, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1036, 1037, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1066, 1067, 1068, 1069, 1070, 1071, 1073, 1074, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112] # for [0.5 0.  0.  0.5 0. ]
        # label = [1, 2, 5, 6, 8, 9, 10, 12, 13, 15, 16, 19, 20, 21, 25, 26, 27, 30, 31, 34, 35, 36, 40, 42, 43, 45, 48, 50, 51, 53, 57, 61, 67, 68, 69, 70, 71, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 93, 94, 97, 98, 101, 102, 104, 107, 108, 109, 110, 113, 115, 118, 119, 120, 124, 125, 126, 127, 128, 130, 131, 140, 143, 144, 146, 148, 149, 150, 151, 152, 154, 155, 158, 159, 162, 165, 167, 168, 171, 175, 178, 181, 188, 193, 195, 196, 198, 202, 204, 209, 212, 214, 215, 217, 218, 219, 221, 225, 226, 230, 231, 233, 235, 238, 240, 242, 243, 250, 251, 254, 258, 259, 262, 265, 268, 269, 272, 280, 285, 287, 288, 290, 291, 295, 298, 305, 310, 311, 312, 316, 317, 318, 319, 320, 321, 322, 324, 328, 329, 330, 332, 333, 340, 342, 343, 345, 346, 347, 349, 350, 351, 352, 353, 354, 355, 356, 358, 364, 365, 366, 368, 369, 370, 371, 372, 373, 375, 378, 382, 384, 387, 388, 389, 390, 393, 395, 398, 400, 401, 402, 406, 407, 408, 411, 412, 413, 414, 415, 417, 419, 421, 423, 424, 425, 426, 427, 428, 429, 430, 433, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451, 452, 455, 456, 457, 463, 464, 465, 468, 471, 472, 473, 474, 476, 477, 479, 481, 482, 484, 489, 490, 494, 496, 497, 498, 500, 501, 503, 506, 507, 509, 512, 513, 514, 515, 516, 517, 518, 522, 523, 525, 528, 529, 530, 532, 534, 535, 536, 538, 539, 540, 542, 545, 546, 548, 549, 550, 551, 554, 556, 558, 559, 561, 563, 565, 567, 568, 570, 571, 572, 573, 577, 578, 579, 580, 581, 583, 584, 588, 589, 598, 599, 602, 603, 604, 606, 607, 610, 612, 613, 615, 618, 619, 620, 622, 623, 625, 626, 627, 629, 630, 632, 633, 636, 637, 638, 640, 646, 647, 648, 649, 650, 651, 653, 656, 657, 659, 661, 663, 669, 670, 671, 672, 674, 676, 677, 682, 684, 685, 689, 690, 691, 692, 693, 695, 698, 701, 703, 704, 705, 706, 708, 714, 717, 720, 721, 723, 724, 728, 729, 731, 732, 733, 735, 738, 741, 743, 744, 749, 753, 755, 756, 757, 761, 763, 767, 769, 770, 771, 773, 774, 775, 776, 782, 789, 790, 791, 794, 796, 798, 802, 804, 806, 807, 808, 809, 814, 825, 826, 827, 830, 831, 836, 842, 843, 846, 848, 851, 852, 853, 856, 860, 861, 864, 865, 866, 867, 868, 869, 872, 879, 880, 881, 887, 888, 889, 891, 892, 893, 897, 900, 902, 903, 904, 907, 910, 911, 912, 914, 916, 918, 919, 920, 923, 926, 929, 930, 932, 933, 936, 939, 940, 944, 945, 947, 948, 952, 953, 954, 957, 958, 960, 963, 965, 967, 969, 971, 972, 973, 974, 976, 977, 978, 984, 988, 994, 996, 997, 998, 999, 1001, 1006, 1007, 1009, 1013, 1014, 1015, 1016, 1019, 1021, 1023, 1025, 1026, 1029, 1037, 1038, 1042, 1043, 1045, 1049, 1050, 1055, 1056, 1057, 1058, 1059, 1060, 1062, 1064, 1066, 1068, 1069, 1076, 1077, 1078, 1079, 1081, 1084, 1085, 1086, 1087, 1090, 1092, 1094, 1096, 1098, 1099, 1105, 1107, 1109, 1110, 663] #[0.5, 0. 0, 0.5, 0] for tsne > 0
        # label = [1, 5, 7, 10, 11, 13, 15, 17, 19, 20, 21, 24, 25, 34, 35, 36, 40, 41, 42, 45, 48, 50, 52, 67, 69, 71, 75, 80, 82, 83, 85, 87, 88, 89, 93, 98, 103, 106, 109, 111, 117, 121, 123, 127, 131, 133, 134, 140, 148, 149, 151, 154, 155, 161, 162, 164, 165, 166, 167, 170, 172, 174, 178, 181, 183, 189, 190, 192, 193, 195, 197, 200, 201, 202, 203, 204, 205, 206, 212, 214, 216, 219, 220, 222, 225, 226, 229, 235, 238, 243, 247, 251, 253, 255, 259, 263, 268, 269, 274, 280, 287, 289, 290, 293, 295, 305, 307, 309, 311, 316, 318, 319, 323, 327, 329, 330, 335, 339, 340, 347, 348, 349, 350, 351, 352, 353, 362, 364, 365, 366, 368, 369, 372, 373, 374, 377, 378, 380, 383, 384, 385, 387, 394, 396, 398, 400, 401, 402, 406, 408, 409, 411, 412, 415, 419, 423, 424, 428, 429, 430, 434, 436, 437, 443, 446, 447, 448, 450, 453, 458, 462, 465, 466, 469, 470, 471, 473, 476, 481, 485, 488, 490, 491, 493, 496, 497, 498, 500, 505, 512, 513, 514, 516, 517, 518, 524, 525, 526, 528, 532, 535, 537, 539, 542, 543, 545, 546, 547, 550, 551, 553, 554, 556, 558, 567, 570, 571, 572, 575, 581, 583, 585, 587, 588, 589, 591, 596, 603, 604, 606, 607, 609, 613, 614, 615, 616, 617, 620, 623, 625, 630, 632, 633, 638, 639, 641, 644, 645, 647, 649, 650, 652, 653, 655, 656, 658, 660, 661, 664, 665, 669, 670, 672, 674, 676, 677, 678, 680, 681, 682, 684, 686, 687, 688, 689, 690, 692, 694, 695, 697, 698, 700, 701, 702, 703, 705, 706, 707, 710, 711, 712, 718, 720, 721, 724, 726, 728, 729, 731, 733, 738, 739, 740, 742, 743, 744, 745, 747, 748, 751, 752, 753, 754, 757, 758, 759, 761, 763, 764, 765, 766, 768, 769, 771, 774, 778, 779, 782, 783, 784, 786, 789, 790, 791, 795, 797, 798, 803, 804, 806, 807, 808, 814, 815, 816, 817, 818, 820, 824, 829, 830, 831, 834, 840, 841, 842, 843, 844, 845, 846, 849, 851, 852, 858, 860, 862, 866, 867, 868, 869, 870, 871, 876, 878, 879, 881, 885, 886, 890, 892, 893, 894, 895, 896, 898, 899, 900, 901, 903, 905, 906, 909, 910, 913, 920, 921, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 939, 941, 942, 945, 946, 949, 950, 952, 953, 956, 957, 958, 959, 962, 963, 964, 966, 969, 970, 972, 974, 975, 976, 977, 978, 980, 981, 983, 984, 985, 986, 989, 991, 992, 993, 995, 997, 999, 1000, 1001, 1002, 1006, 1009, 1011, 1012, 1013, 1014, 1022, 1023, 1024, 1025, 1027, 1030, 1031, 1033, 1037, 1040, 1041, 1042, 1043, 1044, 1046, 1054, 1055, 1057, 1058, 1060, 1061, 1064, 1065, 1067, 1068, 1069, 1075, 1077, 1078, 1079, 1080, 1081, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1093, 1095, 1096, 1098, 1099, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1110, 1111, 1112, 510
# ] [0, 0.5, 0, 0.5, 0] pos > 0
#         label = [0, 3, 5, 6, 11, 17, 18, 20, 26, 27, 28, 29, 31, 32, 37, 38, 46, 47, 49, 53, 55, 59, 60, 62, 64, 65, 72, 73, 75, 77, 78, 84, 85, 90, 91, 94, 95, 96, 99, 107, 108, 116, 120, 121, 122, 123, 128, 130, 132, 138, 141, 142, 143, 145, 147, 148, 149, 153, 154, 156, 159, 162, 164, 169, 171, 173, 175, 179, 181, 182, 184, 185, 186, 187, 188, 190, 199, 203, 207, 209, 210, 212, 223, 224, 227, 228, 231, 233, 234, 235, 237, 238, 239, 240, 248, 249, 251, 252, 254, 258, 260, 263, 265, 266, 270, 274, 275, 276, 277, 278, 280, 282, 284, 285, 287, 289, 290, 296, 299, 321, 324, 325, 333, 336, 337, 342, 343, 347, 350, 354, 356, 357, 358, 359, 360, 361, 362, 363, 365, 367, 371, 380, 386, 391, 392, 393, 396, 397, 399, 403, 405, 407, 409, 410, 412, 417, 418, 431, 432, 435, 438, 440, 441, 450, 451, 457, 459, 460, 467, 468, 470, 472, 479, 484, 486, 488, 492, 493, 496, 497, 498, 499, 503, 504, 508, 509, 513, 520, 522, 531, 533, 537, 541, 542, 544, 547, 550, 552, 559, 560, 561, 562, 567, 569, 574, 575, 577, 579, 580, 582, 584, 586, 592, 593, 594, 597, 600, 606, 608, 618, 624, 626, 631, 634, 637, 640, 645, 649, 654, 661, 665, 666, 667, 669, 673, 676, 678, 682, 684, 686, 692, 694, 696, 699, 700, 701, 702, 705, 706, 707, 709, 711, 712, 713, 714, 718, 719, 725, 726, 729, 730, 733, 734, 736, 739, 740, 742, 743, 746, 747, 748, 749, 751, 752, 755, 757, 758, 760, 762, 763, 764, 765, 766, 768, 771, 774, 777, 778, 779, 780, 786, 787, 788, 789, 790, 793, 794, 795, 797, 798, 799, 800, 801, 802, 805, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 818, 821, 823, 824, 829, 830, 831, 833, 834, 835, 839, 842, 846, 847, 849, 850, 851, 855, 858, 862, 863, 864, 866, 867, 872, 873, 875, 876, 878, 879, 884, 886, 888, 890, 892, 894, 895, 899, 905, 906, 909, 915, 920, 923, 926, 927, 928, 930, 931, 933, 935, 936, 937, 938, 942, 947, 949, 950, 955, 958, 959, 961, 964, 968, 969, 972, 975, 976, 979, 980, 982, 983, 985, 988, 989, 990, 991, 993, 996, 997, 1000, 1002, 1005, 1006, 1008, 1010, 1011, 1012, 1013, 1014, 1016, 1018, 1022, 1024, 1025, 1026, 1029, 1030, 1033, 1034, 1036, 1039, 1040, 1041, 1042, 1044, 1046, 1049, 1051, 1052, 1053, 1054, 1055, 1057, 1058, 1061, 1064, 1065, 1068, 1070, 1072, 1073, 1074, 1077, 1078, 1079, 1080, 1081, 1084, 1086, 1087, 1088, 1091, 1093, 1095, 1096, 1100, 1101, 1102, 1104, 1105, 1106, 1108, 1111, 1112, 510
#         ] # [0, 0.5, 0, 0.5, 0] abs(pos) >0.1
        label = [5, 11, 13, 15, 17, 20, 21, 24, 45, 50, 67, 69, 75, 80, 85, 87, 93, 103, 121, 123, 127, 148, 149, 154, 161, 162, 164, 165, 167, 170, 172, 174, 178, 181, 189, 190, 192, 200, 201, 202, 203, 204, 205, 212, 220, 225, 226, 229, 235, 238, 247, 251, 255, 263, 268, 274, 280, 287, 289, 290, 305, 311, 329, 340, 347, 350, 351, 352, 362, 365, 366, 377, 378, 380, 384, 394, 396, 400, 401, 402, 408, 409, 412, 415, 419, 424, 429, 443, 446, 450, 453, 462, 466, 470, 473, 476, 481, 485, 488, 490, 493, 496, 497, 498, 505, 512, 513, 516, 518, 526, 528, 532, 537, 539, 542, 543, 546, 547, 550, 554, 567, 571, 572, 575, 591, 606, 609, 615, 620, 625, 630, 632, 645, 649, 653, 655, 656, 661, 665, 669, 672, 674, 676, 678, 682, 684, 686, 687, 692, 694, 697, 700, 701, 702, 705, 706, 707, 710, 711, 712, 718, 726, 729, 731, 733, 738, 739, 740, 742, 743, 747, 748, 751, 752, 757, 758, 763, 764, 765, 766, 768, 771, 774, 778, 779, 783, 784, 786, 789, 790, 791, 795, 797, 798, 803, 807, 808, 814, 815, 816, 818, 820, 824, 829, 830, 831, 834, 841, 842, 844, 846, 849, 851, 858, 862, 866, 867, 868, 876, 878, 879, 881, 885, 886, 890, 892, 893, 894, 895, 899, 905, 906, 909, 910, 920, 921, 926, 927, 928, 929, 930, 931, 932, 933, 935, 939, 941, 942, 946, 949, 950, 956, 958, 959, 962, 963, 964, 969, 970, 972, 974, 975, 976, 978, 980, 981, 983, 984, 985, 989, 991, 993, 995, 997, 999, 1000, 1001, 1002, 1006, 1011, 1012, 1013, 1014, 1022, 1024, 1025, 1030, 1033, 1037, 1040, 1041, 1042, 1044, 1046, 1054, 1055, 1057, 1058, 1061, 1064, 1065, 1067, 1068, 1069, 1075, 1077, 1078, 1079, 1080, 1081, 1084, 1086, 1087, 1088, 1093, 1095, 1096, 1101, 1102, 1103, 1104, 1105, 1106, 1108, 1110, 1111, 1112
                ]  #[0, 0.5, 0, 0.5, 0] pos > 0
        label = [5, 11, 13, 15, 17, 20, 21, 24, 45, 50, 67, 69, 75, 80, 85, 87, 93, 103, 121, 123, 127, 148, 149, 154,
                  161, 162, 164, 165, 167, 170, 172, 174, 178, 181, 189, 190, 192, 200, 201, 202, 203, 204, 205, 212,
                  220, 225, 226, 229, 235, 238, 247, 251, 255, 263, 268, 274, 280, 287, 289, 290, 305, 311, 329, 340,
                  347, 350, 351, 352, 362, 365, 366, 377, 378, 380, 384, 394, 396, 400, 401, 402, 408, 409, 412, 415,
                  419, 424, 429, 443, 446, 450, 453, 462, 466, 470, 473, 476, 481, 485, 488, 490, 493, 496, 497, 498,
                  505, 512, 513, 516, 518, 526, 528, 532, 537, 539, 542, 543, 546, 547, 550, 554, 567, 571, 572, 575,
                  591, 606, 609, 615, 620, 625, 630, 632, 645, 649, 653, 655, 656, 661, 665, 669, 672, 674, 676, 678,
                  682, 684, 686, 687, 692, 694, 697, 700, 701, 702, 705, 706, 707, 710, 711, 712, 718, 726, 729, 731,
                  733, 738, 739, 740, 742, 743, 747, 748, 751, 752, 757, 758, 763, 764, 765, 766, 768, 771, 774, 778,
                  779, 783, 784, 786, 789, 790, 791, 795, 797, 798, 803, 807, 808, 814, 815, 816, 818, 820, 824, 829,
                  830, 831, 834, 841, 842, 844, 846, 849, 851, 858, 862, 866, 867, 868, 876, 878, 879, 881, 885, 886,
                  890, 892, 893, 894, 895, 899, 905, 906, 909, 910, 920, 921, 926, 927, 928, 929, 930, 931, 932, 933,
                  935, 939, 941, 942, 946, 949, 950, 956, 958, 959, 962, 963, 964, 969, 970, 972, 974, 975, 976, 978,
                  980, 981, 983, 984, 985, 989, 991, 993, 995, 997, 999, 1000, 1001, 1002, 1006, 1011, 1012, 1013, 1014,
                  1022, 1024, 1025, 1030, 1033, 1037, 1040, 1041, 1042, 1044, 1046, 1054, 1055, 1057, 1058, 1061, 1064,
                  1065, 1067, 1068, 1069, 1075, 1077, 1078, 1079, 1080, 1081, 1084, 1086, 1087, 1088, 1093, 1095, 1096,
                  1101, 1102, 1103, 1104, 1105, 1106, 1108, 1110, 1111, 1112
                  ]  # [0, 0.5, 0, 0.5, 0] pos > 0

        label_exclude = [ 366 , 981 , 485 , 443 , 247 , 672 , 45 , 526 , 702 , 311 , 512 , 351 , 609 , 546 , 466 , 881 , 200 , 165 , 571 , 632 , 1112 , 687 , 993 , 625 , 255 , 453 , 731 , 766 , 1065 , 50 , 172 , 378 , 167 , 665 , 740 , 991 , 927 , 127 , 462 , 402 , 161 , 939 , 17 , 170 , 1086 , 178 , 415 , 220 , 518 , 758 , 497 , 516 , 401 , 1075 , 1110 , 893 , 738 , 941 , 225 , 419 , 532 , 841 , 820 , 400 , 473 , 783 , 13 , 164 , 844 , 329 , 498 , 93 , 69 , 340 ,
        189 , 394 , 999 , 539 , 694 , 1001 , 24 , 591 , 886 , 528 , 226 , 488 , 554 , 1069 , 80 , 11 , 778 , 148 , 739 , 963 , 123 , 377 , 201 , 816 , 909 , 791 , 868 , 543 , 697 , 910 , 174, 362 , 103 , 15 , 384 , 1024 , 5 , 932 , 752 , 190 , 692 , 154 , 181 , 493 , 742 , 20 , 263 , 962 , 757 , 496 , 235 , 424 , 229 , 21 , 238 , 656 , 205 , 476 , 620 , 878 , 1011 , 305 , 984 , 572 , 490 , 1037 , 653 , 1103 , 803 , 347 , 446 , 290 , 1006 , 645 , 429 , 701 , 274 , 287 , 121 , 149 , 829 ,
        931 , 537 , 684,  989 , 251 , 831 , 956 , 974 , 712 , 505 , 784 , 408 , 764 , 894 , 674 , 970 , 790 , 995 , 795 , 550 , 470 , 710 , 1067 , 814 , 212 , 771 , 67 , 280 , 1102 , 87]
        # label = [i for i in label if i not in label_exclude]
        print label
    return label


def embedding_plot(pos, tsne_pos, title_cache, y_color, y, annotate):
    pos = np.array(pos)
    y_color = np.array(y_color)
    assert np.shape(pos)[1]==2
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    print('Type of pos is %s. Shape of pos is %s'%(type(pos), np.shape(pos)))
    # ax.scatter(pos, c=y_color, s=2)
    ax.scatter(np.array(pos[:, 0]), np.array(pos[:, 1]), c=y_color, s =2) # s = 2
    (beta, graph, round, train_acc, test_acc,c, sigma) = title_cache
    from textwrap import wrap
    title_text = str(beta) + ' ' + graph + ' Round ' + str(round) + 'SVM:Train:' + str(train_acc) + ' Test:' + str(
        test_acc) + ' C: ' + str(c) + ' Sigma: ' + str(sigma)
    title = ax.set_title("\n".join(wrap(title_text, 60)))
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)

    import matplotlib.pyplot as plt
    for xi, yi, pidi in zip(pos[:, 0], pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    ax = fig.add_subplot(212)
    ax.scatter(tsne_pos[:, 0], tsne_pos[:, 1], c=y_color, s=1.2)
    for xi, yi, pidi in zip(tsne_pos[:, 0], tsne_pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())
    return fig
def MDS(dist_matrix, y, cache,rs=42, annotate='no', print_flag='False', gd='False'):
    # input is the distance matrix
    # ouput: draw the mds/tsne 2D embedding

    graphsimport()
    np.set_printoptions(precision=3)
    # random.seed(rs)
    assert np.shape(dist_matrix)[0] == np.shape(dist_matrix)[1] == len(y)

    (beta, graph, round, train_acc, test_acc, c, sigma) = cache
    train_acc = (100 * train_acc).round()
    test_acc = (100 * test_acc).round()
    cache = (beta, graph, round, train_acc, test_acc, c, sigma)

    start_time = timeit.default_timer()
    mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
    tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
    pos = mds.fit_transform(dist_matrix)
    tsne_pos = tsne.fit_transform(dist_matrix)
    print('Computing MDS and SNE takes %s'%(timeit.default_timer() - start_time))

    print('Right part:')
    for i in range(np.shape(pos)[0]):
        # continue
        if ((pos[i,0] > 0.0) ) and (print_flag=='True'):
            print('%s,'%i),

    # print(np.shape(pos))
    # y = change_label(y, new_label('mutag'))
    # print(list(y).count(1))
    y_color = [color_map(i) for i in y]
    assert len(y_color) == len(y)
    print(y_color.count('r'))
    assert np.shape(pos)[0] == len(y); assert np.shape(tsne_pos)[0] == len(y)

    fig = embedding_plot(pos, tsne_pos, cache, y_color, y, annotate)

    direct = './Viz_algorithm/' + graph + '/distance_matrix/'; make_direct(direct)
    if gd == 'True':
        direct = './Viz_algorithm/' + graph + '/gd/'; make_direct(direct)
    filename = str(round) + '_mds_' + str(beta) + '.png'
    fig.savefig(direct + filename); print('Saving figure successfully')
def viz_persistence_vector(dgms, Y, graph, beta, X, rf, X_flag='No'):
    # X_flag
    import time
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as mp

    start = time.time()
    for i in range(0, len(dgms)):

        if Y[i]==1:
            color = 'skyblue'
        elif Y[i]==-1:
            color = 'lightcoral'
        elif Y[i] == 0:
            color = 'lightgreen'
        elif Y[i] == 2:
            color = 'palegoldenrod'
        else:
            print('Not color set for %s'%Y[i])
            return
        if X_flag=='No':
            distr = persistence_vector(dgms[i])
            mp.plot(distr[0], c=color, alpha=1, linewidth=.4)
        elif X_flag=='Yes': # visualize X directly instead of computing X from dgms
            mp.plot(X[i], c=color, alpha=.5, linewidth=.5)
        # mp.ylim(0,10)
    mp.title(graph + ' ' + str(beta) + '\nrf:' + str(round(100 * rf[1])) + 'BL:' + str(baseline(graph)))
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/vector/persistence_vector/'
    make_direct(direct)
    mp.savefig(direct + str(beta), format='svg')
    mp.close()
    print('viz_persistence_vector takes %s'%(time.time()-start))
    # mp.show()

import matplotlib.pyplot as plt
import numpy as np
print('i am here')
# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
plt.figure()
print('start drawing...')
plt.boxplot(data)
plt.show()
import sys
sys.exit()

spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
data.shape = (-1, 1)
d2.shape = (-1, 1)
data = [data, d2, d2[::2, 0]]
# multiple box plots on one figure
plt.figure()
plt.boxplot(data)
plt.show()