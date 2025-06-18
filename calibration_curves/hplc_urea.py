import pandas as pd
import dill
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def generate_calibration_curve_spreadsheet():
    pass


def identity(x):
    return x


def log10(x):
    return np.log10(x)

def error_log_log_fit(results, dy):
    """Returns the error on a point if the curve was originally fit as log-log"""
    return dy / y

def error_linear_linear_fit(dy):
    """Returns the error on a point if the curve was originally fit as log-log"""
    return dy

def get_y_uncertainty_function(results, X, y_transformation, x_transformation, add_constant_flag):
    """Returns the proper function to calculate the y uncertainty of a point"""

    dy = np.sqrt(results.mse_resid) # get the uncertainty on the fit
    X = x_transformation(X) # transform X appropriately for the fit
    if add_constant_flag:
        X = sm.add_constant(X)

    if y_transformation == identity or y_transformation is None:
        def y_uncertainty_function(X):
            dC = dy
            return dC
        return error_linear_linear_fit
    elif y_transformation == log10 or y_transformation == np.log10:
        def y_uncertainty_function(X):
            K = np.log(10.)
            dC = np.abs(dy * K * results.predict(X))
            return dC
    else:
        print('In get_y_uncertainty. Transformation function not recognized.')
        print('Error will return all zeros.')
        def y_uncertainty_function(X):
            return 0 * X

    return y_uncertainty_function


    return

def filter_data(p: Path, x, y, x_range, x_transformation, y_transformation, add_X_constant):
    if p.suffix == '.csv':
        df = pd.read_csv(p)
    elif p.suffix == '.xlsx':
        df = pd.read_excel(p)
    else:
        print('Wrong data type')
        return

    if x_range is None:
        pass
    else:
        print('Filtering over {}, {}'.format(x_range[0], x_range[1]))
        df = df.where(df[x] > x_range[0]).where(df[x] < x_range[1]).dropna(how='all')

    y = y_transformation(df[y])
    X = x_transformation(df[x])
    if add_X_constant:
        X = sm.add_constant(X)
    return df, X, y


def fit_calibration_curve(p: Path, x='area', y='concentration', x_transformation=identity, y_transformation=identity,
                          x_transformation_label='identity', y_transformation_label='identity', add_constant_flag=True,
                          x_range=None,
                          save_to=Path('./results.dill'),
                          print_flag=False):
    df = filter_data(p, x, y, x_range)

    y = y_transformation(df[y])
    X = x_transformation(df[x])
    if add_constant_flag:
        print('blag blah blah')
        X = sm.add_constant(X)
    print('X:\n{}'.format(X))

    model = sm.OLS(y, X)
    results = model.fit()
    dy_function = get_y_uncertainty_function(results, X, y_transformation, x_transformation, add_constant_flag)


    y_function = get_loglog_y_function(results.params[0], results.params[1])



    if print_flag:
        print('Results from: {}'.format(p.name))
        print('X-transformation: {}'.format(x_transformation_label))
        print(results.summary())
        # print('dy: {}'.format(dy))

    with open(save_to, 'wb') as outfile:
        dill.dump(results, outfile)

    uncertainty_path = save_to.parent / (save_to.stem + '_uncertainty')
    # uncertainty_path = uncertainty_path.with_suffix('.dill')
    with open(uncertainty_path, 'wb') as outfile:
        dill.dump(dy_function,outfile)
        
    summary_fp = save_to.parent / (save_to.stem + '_summary.txt')
    with open(summary_fp, 'w') as outfile:

        if x_range is None:
            outfile.write('x range: ({},{})\n'.format(df[x].min(), df[x].max()))
        else:
            outfile.write('x range: ({},{})\n'.format(x_range[0], x_range[1]))

        outfile.write('d(prediction): {}\n'.format(np.sqrt(results.mse_resid)))
        outfile.write('x transformation: {}\n'.format(x_transformation_label))
        outfile.write('y transformation: {}\n'.format(y_transformation_label))
        outfile.write('\n\n')
        outfile.write(str(results.summary()))

        # help(results.summary
    return y_function, dy_function




def get_power_law_y_function(a0, a1):
    def y(x):
        return 10**a0 * x**a1
    return y

def get_power_law_dy_function(y_func, dlogy):
    def dy(x):
        y = y_func(x)
        uncertainty = np.abs(y * np.log(10.)) * dlogy
        return uncertainty
    return dy


def powerlawfit(p: Path, x='area', y='concentration', add_X_constant=True, x_range=None, y_transformation=np.log, x_transformation=np.log
                ):
    df, X, y = filter_data(p, x, y, x_range, x_transformation, y_transformation, add_X_constant)

    model = sm.OLS(y,X)
    res = model.fit()

    y_func = get_power_law_y_function(res.params.iloc[0], res.params.iloc[1])
    dlogy = np.sqrt(res.mse_resid) # get the uncertainty on the fit
    percent_err = 100 * dlogy * np.log(10.)

    dy_func = get_power_law_dy_function(y_func, dlogy)
    return y_func, dy_func, res


def get_exponential_y_function(a0, a1):
    def y(x):
        return a0 * (np.e ** (a1 * x))
    return y

def get_exponential_dy_function(y_func, dlogy):
    def dy(x):
        y = y_func(x)
        uncertainty = y * dlogy
        return uncertainty
    return dy

def exponentialfit(p: Path, x='area', y='concentration', add_X_constant=True, x_range=None,  y_transformation=np.log, x_transformation=identity):
    """Log base e fit"""
    df, X, y = filter_data(p, x, y, x_range, x_transformation, y_transformation, add_X_constant)

    model = sm.OLS(y,X)
    res = model.fit()

    y_func = get_exponential_y_function(res.params.iloc[0], res.params.iloc[1])
    dlogy = np.sqrt(res.mse_resid)
    dy_func = get_exponential_dy_function(y_func, dlogy)
    return y_func, dy_func, res


def get_linear_y_function(a0, a1):
    def y(x):
        return a0 + a1 * x
    return y


def get_linear_dy_function(dy_val):
    def dy(x):
        return dy_val
    return dy


def linearfit(p: Path, x='area', y='concentration', add_X_constant=True, x_range=None,  y_transformation=identity, x_transformation=identity):
    """Log base e fit"""
    df, X, y = filter_data(p, x, y, x_range, x_transformation, y_transformation, add_X_constant)

    model = sm.OLS(y,X)
    res = model.fit()

    y_func = get_linear_y_function(res.params.iloc[0], res.params.iloc[1])
    
    dy = np.sqrt(res.mse_resid)
    dy_func = get_linear_dy_function(dy)
    return y_func, dy_func, res


def taylor_exponential_fit_example():
    ### this agrees with what is found in Taylor, page 195: 11.93 and -0.089
    p = Path('taylor_exponential_example.xlsx')
    df = pd.read_excel(p)
    y, dy, res = exponentialfit(p, x='x', y='y')
    print(res.summary())
    return


def save_model(path_out: Path, y_function, dy_function, fitting_result: sm.OLS):
    func_path = path_out.with_suffix('.y')
    err_path = path_out.with_suffix('.dy')
    mod_path = path_out.with_suffix('.model')

    def save(p, obj):
        with open(p, 'wb') as outfile:
            # help(dill.dump)
            dill.dump(obj, outfile)
        return

    save(func_path, y_function)
    save(err_path, dy_function)
    save(mod_path, fitting_result)
    return


def load_model(path_in):
    func_path = path_in.with_suffix('.y')
    print(func_path)
    err_path = path_in.with_suffix('.dy')
    mod_path = path_in.with_suffix('.model')

    def load(p):
        with open(p, 'rb') as infile:
            obj = dill.load(infile)
        return obj

    y = load(func_path)
    dy = load(err_path)
    model = load(mod_path)
    return y, dy, model

def main():

    p = Path('/Users/ianbillinge/Documents/kimlab/projects/vuv/xanthydrol/')
    infile_path = p / '20241211 HPLC Urea-Xan.xlsx'
    save_to_path = p / '2024_12_11'

    ### fit models
    powerlaw_y, powerlaw_dy, powerlaw_model = powerlawfit(infile_path)
    exp_y, exp_dy, exponential_model = exponentialfit(infile_path)
    lin_y, lin_dy, linear_model = linearfit(infile_path)
    lin_low_y, lin_low_dy, linear_low_model = linearfit(infile_path, x_range=(0,200))
    lin_high_y, lin_high_dy, linear_high_model = linearfit(infile_path, x_range=(100, 10000))

    path_out_lin_low = p / 'fits' / '2025_06_17_low'
    save_model(path_out_lin_low, lin_low_y, lin_low_dy, linear_low_model)

    xx = 'area'
    yy = 'concentration'
    df, x, y = filter_data(infile_path, x=xx, y=yy,x_range=None, x_transformation=identity, y_transformation=identity, add_X_constant=True)
    df_log, x_log, y_log = filter_data(infile_path, x=xx, y=yy, x_range=None, x_transformation=log10, y_transformation=log10, add_X_constant=True)

    def plot_results():
        fig, ax = plt.subplots(nrows=2)

        ### Plot
        x = df[xx]
        y = df[yy]
        x1 = np.linspace(x.min(), x.max(), 30)
        print(x1,lin_y(x1), lin_dy(x1))
        ax[0].plot(df_log[xx], df_log[yy], 'o', label='real data')
        ax[0].errorbar(x,lin_y(x), yerr=lin_dy(x), label='linear model')
        ax[0].errorbar(x,lin_low_y(x), yerr=lin_low_dy(x), label='linear low model')
        ax[0].errorbar(x,lin_high_y(x), yerr=lin_high_dy(x), label='linear high model')


        ax[1].plot(x,y, 'o',label='real data')
        ax[1].plot
        ax[1].errorbar(x1, powerlaw_y(x1), yerr=powerlaw_dy(x1),label='power law prediction')
        ax[1].errorbar(x1, exp_y(x1), yerr=exp_dy(x1),label='exponential prediction')
        [axis.legend() for axis in ax]

        plt.show()

    # plot_results()

 
    return


if __name__ == '__main__':
    main()
