import os


def parser_transcription(filename):
    """parse the transcription"""

    onset, offset, label = [], [], []
    with open(filename, 'r') as trans_file:
        content = trans_file.readlines()
        for line in content:
            line = line.split()
            onset.append(float(line[0]))
            offset.append(float(line[1]))
            label.append(line[2])
    return onset, offset, label


def onset_offset_label_yield(filepath_nus, read_sing):
    """yield onset, offset and label for each singing recording"""
    subpaths = os.listdir(filepath_nus)
    for sp in subpaths:
        if sp != 'README.txt':
            path_read_sing = os.path.join(filepath_nus, sp, read_sing)
            for fn in os.listdir(path_read_sing):
                if '.txt' in fn:
                    fn_trans = os.path.join(filepath_nus, sp, read_sing, fn)
                    onset, offset, label = parser_transcription(fn_trans)
                    yield fn, sp, onset, offset, label

