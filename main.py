import cv2 as cv
import numpy as np

offset_to_use = 4
margins = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
sets_games = [1, 2, 3, 4, 5] #modify this to test on different sets of games, in training folder there are 5 sets of games

empty_board = [
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
    ]

letters_dict = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8, 'J' : 9, 'K' : 10, 'L' : 11, 'M' : 12, 'N' : 13, 'O' : 14}

traseu = [-1,1,2,3,4,5,6,0,2,5,3,4,6,2,2,0,3,5,4,1,6,2,4,5,5,0,6,3,4,2,0,1,5,1,3,4,4,4,5,0,6,3,5,4,1,3,2,0,0,1,1,2,3,6,3,5,2,1,0,6,6,5,2,1,2,5,0,3,3,5,0,6,1,4,0,6,
          3,5,1,4,2,6,2,3,1,6,5,6,2,0,4,0,1,6,4,4,1,6,6,3,0]



def detect_circles(image):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 10,
                              param1=50, param2=11, minRadius=8, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return len(circles[0])
    else:
        return None


def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(image, margin=3):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_blue = np.array([40, 120, 20])
    upper_blue = np.array([143, 255, 255])
    mask = cv.inRange(hsv_image, lower_blue, upper_blue)
    result = cv.bitwise_and(hsv_image, hsv_image, mask=mask)
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    # apply gaussian blur, median blur and threshold
    image_g_blur = cv.GaussianBlur(result, (5, 5), 0)
    image_m_blur = cv.medianBlur(result, 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)

    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=8)
    thresh = cv.dilate(thresh, kernel)

    edges = cv.Canny(thresh, 100, 400)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    top_left = (max(top_left[0] - margin, 0), max(top_left[1] - margin, 0))
    bottom_right = (min(bottom_right[0] + margin, image.shape[1]), min(bottom_right[1] + margin, image.shape[0]))
    top_right = (min(top_right[0] + margin, image.shape[1]), max(top_right[1] - margin, 0))
    bottom_left = (max(bottom_left[0] - margin, 0), min(bottom_left[1] + margin, image.shape[0]))

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_HSV2BGR)

    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)

    width = 1350
    height = 1350

    puzzle_corners = np.array([[top_left], [top_right], [bottom_right], [bottom_left]], dtype=np.float32)
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    perspective_transform = cv.getPerspectiveTransform(puzzle_corners, destination_of_puzzle)
    result = cv.warpPerspective(image, perspective_transform, (width, height))
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    return result


def determina_configuratie_careu(thresh, lines_horizontal, lines_vertical):
    global max_patch_i
    global max_patch_j
    global patches
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            medie_patch = np.mean(patch)
            patches.append([medie_patch, i, j])

    patches.sort(key=lambda x: x[0], reverse=True)


for current_set in sets_games:

    matrix_of_pieces = [[False for i in range(15)] for j in range(15)]
    print('Starting set ' + str(current_set) + '...')

    for index in range(1, 21):
        patches = []
        image_index = ''
        if index < 10:
            image_index = '0' + str(index)
        else:
            image_index = str(index)

        image_name = 'testare/' + str(current_set) + '_' + str(image_index) + '.jpg' #change 'antrenare' with the path of your machine where you have the training data
        img = cv.imread(image_name)

        # alpha = 1.2
        # beta = 2
        # img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

        careu = extrage_careu(img)
        img_tabla_goala = cv.imread('imagini_auxiliare/01.jpg')
        result_tabla_goala = extrage_careu(img_tabla_goala)

        _, result = cv.threshold(careu, 175, 255, cv.THRESH_BINARY)
        _, result_tabla_goala = cv.threshold(result_tabla_goala, 190, 255, cv.THRESH_BINARY)
        diferenta = cv.absdiff(result, result_tabla_goala)

        lines_horizontal=[]
        for i in range(0,1351,90):
            l=[]
            l.append((0,i))
            l.append((1349,i))
            lines_horizontal.append(l)


        lines_vertical=[]
        for i in range(0,1351,90):
            l=[]
            l.append((i,0))
            l.append((i,1349))
            lines_vertical.append(l)

        for line in lines_vertical:
            cv.line(result, line[0], line[1], (0, 255, 0), 2)
            for line in lines_horizontal:
                cv.line(result, line[0], line[1], (0, 0, 255), 2)

        determina_configuratie_careu(diferenta,lines_horizontal,lines_vertical)

        for patch_info in patches[:index*2]:
            i_first = patch_info[1]
            j_first = patch_info[2]

            all_circles_detected = []

            for margin in margins:
                careu = extrage_careu(img, margin)
                _, result = cv.threshold(careu, 175, 255, cv.THRESH_BINARY)

                y_min = max(lines_vertical[j_first][0][0] - offset_to_use, 0)
                y_max = min(lines_vertical[j_first + 1][1][0] + offset_to_use, result.shape[1])
                x_min = max(lines_horizontal[i_first][0][1] - offset_to_use, 0)
                x_max = min(lines_horizontal[i_first + 1][1][1] + offset_to_use, result.shape[0])
                patch = result[x_min:x_max, y_min:y_max].copy()

                circles = detect_circles(patch)
                all_circles_detected.append(circles)

            circles = max(all_circles_detected, key=all_circles_detected.count)
            number_of_circles = 0

            # check if the piece was marked in the matrix
            if matrix_of_pieces[i_first][j_first] == False:
                matrix_of_pieces[i_first][j_first] = True

                file_to_write = 'my_tests/' + str(current_set) + '_' + str(image_index) + '.txt'
                g = open(file_to_write, "a")
                if circles is None:
                    number_of_circles = 0
                else:
                    number_of_circles = circles

                piece = str(i_first + 1) + letters[j_first] + ' ' + str(number_of_circles) + '\n'
                print(piece)
                g.write(piece)

    player1_score = 0
    player2_score = 0
    current_score_1 = -1
    current_score_2 = -1

    player1_first = 0
    player2_first = 0

    player1_position = 0
    player2_position = 0

    fisier_mutari = 'testare/' + str(current_set) + '_mutari.txt' #change 'antrenare' with the path of your machine where you have the training data
    mutari = []

    with open(fisier_mutari, 'r') as f:
        lines = f.readlines()
        for line in lines:
            mutari.append(line.split())

    players_order = []
    for mutare in mutari:
        players_order.append(mutare[1])

    turn = 0

    for round in range(1, 21):
        if round < 10:
            file_index = '0' + str(round)
        else:
            file_index = str(round)

        file = 'my_tests/' + str(current_set) + '_' + str(file_index) + '.txt'

        pieces = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pieces.append(line.split())

        piece1 = pieces[0]
        piece2 = pieces[1]

        x_piece1 = int(piece1[0][:len(piece1[0]) - 1]) - 1
        y_piece1 = letters_dict[piece1[0][len(piece1[0]) - 1]]
        x_piece2 = int(piece2[0][:len(piece2[0]) - 1]) - 1
        y_piece2 = letters_dict[piece2[0][len(piece2[0]) - 1]]

        value_piece1 = int(piece1[1])
        value_piece2 = int(piece2[1])

        current_player = players_order[turn]
        turn += 1

        if current_player == 'player1':
            if player1_first == -1:
                player1_first = 1
                pass
            else:
                if traseu[player1_position] == value_piece2 or traseu[player1_position] == value_piece1:
                    player1_score += 3
                    player1_position = player1_position + 3

                if traseu[player2_position] == value_piece2 or traseu[player2_position] == value_piece1:
                    player2_score += 3
                    player2_position = player2_position + 3

                if empty_board[x_piece1][y_piece1] != 0:
                    player1_score += empty_board[x_piece1][y_piece1]
                    player1_position = player1_position + empty_board[x_piece1][y_piece1]

                    if value_piece2 == value_piece1:
                        player1_score += empty_board[x_piece1][y_piece1]
                        player1_position = player1_position + empty_board[x_piece1][y_piece1]

                if empty_board[x_piece2][y_piece2] != 0:
                    player1_score += empty_board[x_piece2][y_piece2]
                    player1_position = player1_position + empty_board[x_piece2][y_piece2]

                    if value_piece2 == value_piece1:
                        player1_score += empty_board[x_piece2][y_piece2]
                        player1_position = player1_position + empty_board[x_piece2][y_piece2]

            current_score_1 = player1_score
            player1_score = 0
            player2_score = 0

        else:
            if player2_first == -1:
                player2_first = 1
                pass
            else:
                if traseu[player1_position] == value_piece2 or traseu[player1_position] == value_piece1:
                    player1_score += 3
                    player1_position = player1_position + 3

                if traseu[player2_position] == value_piece2 or traseu[player2_position] == value_piece1:
                    player2_score += 3
                    player2_position = player2_position + 3

                if empty_board[x_piece1][y_piece1] != 0:
                    player2_score += empty_board[x_piece1][y_piece1]
                    player2_position = player2_position + empty_board[x_piece1][y_piece1]

                    if value_piece2 == value_piece1:
                        player2_score += empty_board[x_piece1][y_piece1]
                        player2_position = player2_position + empty_board[x_piece1][y_piece1]

                if empty_board[x_piece2][y_piece2] != 0:
                    player2_score += empty_board[x_piece2][y_piece2]
                    player2_position = player2_position + empty_board[x_piece2][y_piece2]

                    if value_piece2 == value_piece1:
                        player2_score += empty_board[x_piece2][y_piece2]
                        player2_position = player2_position + empty_board[x_piece2][y_piece2]

            current_score_2 = player2_score
            player2_score = 0
            player1_score = 0

        file_current_round = 'my_tests/' + str(current_set) + '_' + str(file_index) + '.txt'

        with open(file_current_round, 'w') as g:

            if x_piece1 < x_piece2 or (x_piece1 == x_piece2 and y_piece1 < y_piece2):
                g.write(str(piece1[0]) + ' ' + str(piece1[1]) + '\n' + str(piece2[0]) + ' ' + str(piece2[1]) + '\n')
            else:
                g.write(str(piece2[0]) + ' ' + str(piece2[1]) + '\n' + str(piece1[0]) + ' ' + str(piece1[1]) + '\n')

            if (current_player == 'player1'):
                g.write(str(current_score_1))
            else:
                g.write(str(current_score_2))


    print('Finished set ' + str(current_set) + '...')
