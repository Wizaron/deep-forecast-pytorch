import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_error_measures(denormal_y, denormal_predicted):

    mae = np.mean(np.absolute(denormal_y - denormal_predicted))
    rmse = np.sqrt((np.mean((np.absolute(denormal_y - denormal_predicted)) ** 2)))
    nrsme_max_min = 100 * rmse / (denormal_y.max() - denormal_y.min())
    nrsme_mean = 100 * rmse / (denormal_y.mean())

    return mae, rmse, nrsme_max_min, nrsme_mean

def draw_graph_station(dataset, yTest, yTestPred, station, visualise=1, ax=None):

    yTest = yTest[:, station]
    denormalYTest = dataset.denormalize_data(yTest)
    denormalPredicted = dataset.denormalize_data(yTestPred[:, station])

    mae, rmse, nrmse_maxMin, nrmse_mean = get_error_measures(denormalYTest, denormalPredicted)
    print 'Station %s : MAE = %7.7s - RMSE = %7.7s - nrmse_maxMin = %7.7s - nrmse_mean = %7.7s'%(station + 1, mae, rmse, nrmse_maxMin, nrmse_mean)

    if visualise:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(denormalYTest, label='Real', color='blue')
        ax.plot(denormalPredicted, label='Predicted', color='red')
        ax.set_xticklabels(range(0, len(yTest), 100), rotation=40)

    return mae, rmse, nrmse_maxMin, nrmse_mean

def draw_graph_all_stations(output_dir, dataset, n_stations, yTest, yTestPred):
    maeRmse = np.zeros((n_stations, 4))

    for staInd in range(n_stations):
        fig, ax = plt.subplots(figsize=(20, 10))
        maeRmse[staInd] = draw_graph_station(dataset, yTest, yTestPred, staInd, visualise=1, ax=ax)
        plt.xticks(range(0, len(yTest), 100))
        filename = '{}/finalEpoch_{}'.format(output_dir, staInd)
        plt.savefig('{}.png'.format(filename))

    errMean = maeRmse.mean(axis=0)
    print 'OUTPUT : ', maeRmse.mean(axis=0)
