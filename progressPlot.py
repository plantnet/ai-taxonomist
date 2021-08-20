import argparse
import regex as re
try:
    import matplotlib.pyplot as plot
except Exception as e:
    print("Please, pip install matplotlib to plot train's progress")
    raise e


def process(log):
    values = {'Train': {}, 'Test': {}, 'Accuracy': {}, 'LearningRate': {}}
    epoch = 0
    for line in log:
        if line.startswith('Epoch:'):
            words = line.split('\t')
            match = re.match(r'Epoch: \[ *(?P<epoch>\d+)\]\[ *(?P<step>\d+)/(?P<max_step>\d+)\]', words[0])
            epoch = int(match.group('epoch'))
            step = int(match.group('step'))
            train_max_step = int(match.group('max_step'))
            x = float(epoch) + float(step) / float(train_max_step)
            for i in range(1, len(words)):
                match = re.match(
                    r'(?P<name>\S+) +(?P<val>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) +\( *(?P<avg>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)',
                    words[i])
                name = match.group('name')
                val = float(match.group('val'))
                avg = float(match.group('avg'))
                values['Train'].setdefault(name, {}).setdefault('x', []).append(x)
                values['Train'].setdefault(name, {}).setdefault('y', []).append(avg)
        elif line.startswith('Test:'):
            words = line.split('\t')
            match = re.match(r'Test: \[ *(?P<step>\d+)/(?P<max_step>\d+)\]', words[0])
            step = int(match.group('step'))
            test_max_step = int(match.group('max_step'))
            x = float(epoch) + float(step) / float(test_max_step)
            for i in range(1, len(words)):
                match = re.match(
                    r'(?P<name>\S+) +(?P<val>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) +\( *(?P<avg>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)',
                    words[i])
                name = match.group('name')
                val = float(match.group('val'))
                avg = float(match.group('avg'))
                values['Test'].setdefault(name, {}).setdefault('x', []).append(x)
                values['Test'].setdefault(name, {}).setdefault('y', []).append(avg)
        elif line.startswith('LearningRate:'):
            words = line.split(' ')
            epoch = int(words[1])
            val = float(words[2])
            values['LearningRate'].setdefault('x', []).append(epoch)
            values['LearningRate'].setdefault('y', []).append(val)
        elif line.startswith(' * '):
            words = line.split(' ')
            for i in range(2, len(words), 2):
                name = words[i]
                val = float(words[i+1])
                values['Accuracy'].setdefault(name, {}).setdefault('x', []).append(epoch)
                values['Accuracy'].setdefault(name, {}).setdefault('y', []).append(val)
        # else:
        #     print(line)
    return values

def draw(val, args):
    fig, ax1 = plot.subplots()

    ax1.set_xlabel('epochs')

    plot.title('C4C progress')
    ax2 = ax1.twinx()
    colors = [
        "#000000", "#ff0000", "#00ff00", "#0000ff", "#a00000", "#00a000",
        "#400000", "#ffcc00", "#00d9ca", "#6600bf", "#b22d2d", "#997a00", "#263332", "#b073e6", "#cc9999",
        "#a3cc00", "#1d6d73", "#c299cc", "#d93a00", "#738040", "#00c2f2", "#e200f2", "#e59173", "#e6f2b6",
        "#002233", "#4b394d", "#594943", "#57664d", "#408cff", "#d936a3", "#4c1f00", "#44ff00", "#234d8c",
        "#731d4b", "#a65800", "#144d00", "#bfd0ff", "#ff4073", "#ffc480", "#74cc66", "#000733", "#331a1d",
        "#4c3913", "#208053", "#696e8c", "#d9c7a3", "#003322", "#3f1d73"
    ]

    colorIdx = 0
    # Losses
    ax1.set_ylabel('loss', color=colors[colorIdx])
    ax1.tick_params(axis='y', labelcolor=colors[colorIdx], color=colors[colorIdx], direction='in')
    if args.max_loss > 0:
        ax1.set_ylim(top=args.max_loss)
    k1 = 'Accuracy'
    k2 = 'ValLoss'
    if len(val[k1]) and k2 in val[k1].keys() and len(val[k1][k2]):
        ax1.plot(val[k1][k2]['x'], val[k1][k2]['y'], label = k2, color=colors[colorIdx], linestyle='dashed')
        colorIdx = (colorIdx + 1) % len(colors)

    # Accuracy
    ax2.set_ylabel('accuracy', color=colors[colorIdx], loc='center')
    ax2.tick_params(axis='y', labelcolor=colors[colorIdx], color=colors[colorIdx])
    k1 = 'Accuracy'
    if len(val[k1]):
        for k2 in ['Acc@1', 'Acc@5']:
            if k2 in val[k1].keys() and len(val[k1][k2]):
                ax2.plot(val[k1][k2]['x'], val[k1][k2]['y'], label = k1+' '+k2, color=colors[colorIdx])
                colorIdx = (colorIdx + 1) % len(colors)

    # Learning rate
    ax3 = ax1.twinx()
    ax3.set_ylabel('learning rate', color=colors[colorIdx], loc='center', labelpad=-50)
    ax3.tick_params(axis='y', labelcolor=colors[colorIdx], color=colors[colorIdx], direction='in', pad=-30)
    # ax3.tick_params(axis='y', labelcolor=colors[colorIdx], color=colors[colorIdx], direction='out')
    plot.yscale('log')
    if len(val['LearningRate']):
        ax3.plot(val['LearningRate']['x'], val['LearningRate']['y'], label="Learning rate", color=colors[colorIdx],
                 linestyle='dotted')
        colorIdx = (colorIdx + 1) % len(colors)

    # train accuracy, if requested
    if args.train:
        k1 = 'Train'
        if len(val[k1]):
            for k2 in ['Acc@1', 'Acc@5']:
                if k2 in val[k1].keys() and len(val[k1][k2]):
                    ax2.plot(val[k1][k2]['x'], val[k1][k2]['y'], label = k1+' '+k2, color=colors[colorIdx])
                    colorIdx = (colorIdx + 1) % len(colors)

    # ax4 = ax2.twinx()
    # ax4.set_ylabel('\nSpeed', color=colors[colorIdx])
    # ax4.tick_params(axis='y', color=colors[colorIdx], direction='in', labelcolor=colors[colorIdx], rotation=90)
    # ax4.plot(data['Speed']['x'], data['Speed']['y'], label="Speed", color=colors[colorIdx], linestyle='dashdot')
    # colorIdx = (colorIdx + 1) % len(colors)

    ax3.set_yscale('log')
    fig.legend(loc=2)

    #    locs, labels = plt.xticks()

    # print("Epoch", epoch)
    # print('Learning rate', lr['y'][-1])
    # for k in perf.keys():
    #     v = perf[k]['y'][-1]
    #     print(k, v)
    # t = 0
    #
    # # if 'Time' in train.keys():
    # #     t += sum(train['Time']['avg'])/float(len(train['Time']['avg']))
    # # t /= len(train)
    # # print('Train Time', t)
    # # t = 0
    # # if 'Data' in train.keys():
    # #     t += sum(train['Data']['avg'])/float(len(train['Data']['avg']))
    # # print('Data loading', t)
    # # t = 0
    # # if 'Time' in test.keys():
    # #     t+= sum(test['Time']['avg'])/float(len(test['Time']['avg']))
    # # print('Test Time', t)
    #
    # if 'Loss' in train.keys():
    #     print('Training loss', train['Loss']['avg'][-1])
    # if 'Loss' in test.keys():
    #     print('Testing loss', test['Loss']['avg'][-1])
    #
    # if args.max_prec:
    #     print('Best epoch', epoch_best)
    #     if len(perf['Top1']['y']) > epoch_best - 1:
    #         print('Best Top1', perf['Top1']['y'][epoch_best - 1])
    #     if len(perf['Top5']['y']) > epoch_best - 1:
    #         print('Best Top5', perf['Top5']['y'][epoch_best - 1])
    #
    # if 'Duration' in perf.keys():
    #     d = sum(perf['Duration']['y'])
    #     print('Duration avg {avg:.3f}'.format(avg=d / float(len(perf['Duration']['y']))))
    #     print('Total duration {sec:.2f} s (or {hour:.3f} h)'.format(sec=d, hour=d / 3600))

    plot.draw()
    plot.show()


def main():
    parser = argparse.ArgumentParser(description='Pytorch train progress plot')
    parser.add_argument('log', type=argparse.FileType('r'), help='train.py output')
    parser.add_argument('--max-loss', type=float, default=0, help='clamp losses', required=False)
    parser.add_argument('--train', action='store_true', help='Print train stats', required=False)

    args = parser.parse_args()

    # process output
    val = process(log=args.log)

    # plot
    draw(val, args)

    print('bye')


if __name__=="__main__":
    main()
