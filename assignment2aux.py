def visualise(tiles, state):
    height = len(tiles)
    width = len(tiles[0])
    print()
    for i in range(height):
        for j in range(width):
            if not tiles[i][j]:
                print(' ', end='')
            elif tiles[i][j] == (0,):
                print(['\u257a', '\u2579', '\u2578', '\u257b'][state[i * width + j]], end='')
            elif tiles[i][j] == (0, 1):
                print(['\u2517', '\u251b', '\u2513', '\u250f'][state[i * width + j]], end='')
            elif tiles[i][j] == (0, 2):
                print(['\u2501', '\u2503', '\u2501', '\u2503'][state[i * width + j]], end='')
            elif tiles[i][j] == (0, 1, 2):
                print(['\u253b', '\u252b', '\u2533', '\u2523'][state[i * width + j]], end='')
            else:
                print()
                print(f'Unexpected tile representation encountered: {tiles[i][j]}.')
                print('Accepted tile representations are (), (0,), (0, 1), (0, 2), (0, 1, 2).')
                raise ValueError(tiles[i][j])
        print()
    print()
