import logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
filename='ExperimentRecord.log',filemode='w')
logging.info('This is for paper')

