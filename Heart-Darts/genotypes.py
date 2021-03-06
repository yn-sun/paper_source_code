from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [

    # 'avg_pool_1x3',
    # 'avg_pool_1x5',
    # 'avg_pool_1x7',
    # 'max_pool_1x2',
    'max_pool_1x3',
    'max_pool_1x5',
    # 'max_pool_1x7',
    'skip_connect',
    'sep_conv_1x3',
    'sep_conv_1x5',
    'sep_conv_1x9',
    'sep_conv_1x13',
    'sep_conv_1x17',
    'sep_conv_1x27',
    'none',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)
AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_1x27', 1), ('sep_conv_1x9', 0), ('sep_conv_1x27', 0), ('sep_conv_1x9', 1), ('skip_connect', 0),
            ('sep_conv_1x9', 2), ('sep_conv_1x3', 3), ('sep_conv_1x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_1x3', 1), ('max_pool_1x5', 0), ('max_pool_1x5', 1), ('max_pool_1x5', 0), ('max_pool_1x3', 0),
            ('sep_conv_1x27', 2), ('sep_conv_1x13', 3), ('sep_conv_1x13', 4)], reduce_concat=range(2, 6))

DARTS_V2 = Genotype(
    normal=[('sep_conv_1x17', 0), ('sep_conv_1x9', 1), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0),
            ('sep_conv_1x17', 1), ('sep_conv_1x17', 2), ('skip_connect', 1), ('sep_conv_1x13', 5), ('skip_connect', 1),
            ('skip_connect', 0), ('sep_conv_1x5', 6), ('sep_conv_1x9', 6), ('sep_conv_1x5', 1)],
    normal_concat=range(2, 9),
    reduce=[('sep_conv_1x17', 0), ('sep_conv_1x9', 1), ('sep_conv_1x13', 1), ('sep_conv_1x17', 2), ('sep_conv_1x13', 2),
            ('sep_conv_1x13', 1), ('sep_conv_1x17', 0), ('sep_conv_1x13', 3), ('sep_conv_1x17', 0), ('sep_conv_1x9', 5),
            ('skip_connect', 6), ('skip_connect', 4), ('skip_connect', 6), ('skip_connect', 4)],
    reduce_concat=range(2, 9))

DARTS = DARTS_V1
