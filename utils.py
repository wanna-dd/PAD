import os
import sys
import logging
import time


def set_logging_defaults(logdir, args):
    if os.path.isdir(logdir):
        res = input('"{}" exists. Overwrite [Y/n]? '.format(logdir))
        if res == 'n':
            raise Exception('"{}" exists.'.format(logdir))
    else:
        os.makedirs(logdir)

    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(sys.stdout)])

    # log cmdline arguments
    logger = logging.getLogger('main')
    logger.info(' '.join(sys.argv))
    logger.info(args)


def get_terminal_size():
    if os.name == 'nt':  # Windows系统
        return 24, 80  # 默认值（24行和80列）
    else:
        try:
            # 在类Unix系统上获取终端大小
            size = os.popen('stty size', 'r').read().split()
            return int(size[0]), int(size[1])
        except ValueError:
            return 24, 80  # 如果出错，则返回默认值


# 获取终端大小 # term_width = 80
term_height, term_width = get_terminal_size()
TOTAL_BAR_LENGTH = 160.
last_time = time.time()
begin_time = last_time


# 其他代码逻辑...
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int((TOTAL_BAR_LENGTH/3)*current/total)
    rest_len = int((TOTAL_BAR_LENGTH/3) - cur_len) - 2

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total - 1:
        sys.stdout.write('\r')  # 更新进度条
    else:
        sys.stdout.write('\n')  # 进度条结束，换行
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
