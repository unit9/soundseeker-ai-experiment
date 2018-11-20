import timeit

from .vectors import angle_between


def find_music(features_list, all_chunks, all_music):

    start = timeit.default_timer()

    distance_maps = []
    chunks_feats = all_chunks[:, 1:11]

    print('features:', features_list)

    for features in features_list:

        distances = list(map(lambda chunk: angle_between(chunk, features),
                             chunks_feats))

        distance_maps.append({all_chunks[i][0]: (i, d) for i, d
                              in enumerate(distances)})

    sorted_music = all_music

    counts = [20, 10, 1]

    for i, s in enumerate(['chunk1FileName', 'chunk2FileName', 'chunk3FileName']):
        sorted_music.sort(key=lambda x: distance_maps[i][x[s]][1])
        sorted_music = sorted_music[:counts[i]]

    end = timeit.default_timer()

    print('find_music:elapsed: {:.2f}ms'.format((end - start) * 1000))

    chosen_music = sorted_music[0]
    chosen_features = [
        all_chunks[distance_maps[0][chosen_music['chunk1FileName']][0]].tolist(),
        all_chunks[distance_maps[1][chosen_music['chunk2FileName']][0]].tolist(),
        all_chunks[distance_maps[2][chosen_music['chunk3FileName']][0]].tolist(),
    ]

    print('chosen features:', chosen_features)

    return chosen_music['fileName'], chosen_features
