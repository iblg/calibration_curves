import pandas as pd
from pathlib import Path
import datetime

def generate_measurement_spreadsheet(parent_dir: str or Path,
                                    file_name: str,
                                    dims: [str],
                                    common_dims:dict,
                                    make_measurement_directory=False,
                                    dilution=False):

    def handle_make_directory(parent_dir):
        if parent_dir.exists():
            pass
        else:
            print('Parent directory {} does not exist.'.format(parent_dir))
            print('Creating parent directory.')
            parent_dir.mkdir()


        if make_measurement_directory is False:
            print('Not making directory for individual measurement. Parent directory is\n{}'.format(parent_dir))

            pass
        elif make_measurement_directory is True:
            print('Making directory for individual measurement. Parent directory is\n{}'.format(parent_dir))
            dirname = str(datetime.date.today())
            parent_dir = parent_dir / dirname
            parent_dir.mkdir()
        elif isinstance(make_directory, str) or isinstance(make_directory, Path):
            dirname = parent_dir / make_directory
            parent_dir.mkdir()
        else:
            print('Don\'t know how to make a directory for individual measurement')
        return parent_dir

    parent_dir = handle_make_directory(parent_dir)

    def handle_dilution():
        if dilution is False:
            return []
        elif dilution == 'volumetric':
            return ['vol sample', 'vol di']
        elif dilution == 'gravimetric':
            return ['m sample', 'm di']
        else:
            print('Don\'t know how to handle dilution for the measurement spreadsheet')
            return []

    dilution = handle_dilution()

    df = pd.DataFrame()
    for k, v in common_dims.items():
        df[k] = v
        
    for dim in dims: ### add columns with names
        df[dim] = 0

    df['area'] = 0
    for i in dilution:
        df[i] = 0

    sheet_path = (parent_dir / file_name).with_suffix('.xlsx')

    if sheet_path.exists():
        print('\n\nSpreadsheet already exists. Spreadsheet is at\n{}'.format(sheet_path))

        print('Proceeding will overwrite spreadsheet.\n')
        response = input('\nReply y to proceed. Anything else will stop spreadsheet creation.\n')
        if response == 'y':
            pass
        else:
            return

    df.to_excel(sheet_path)
    return


def main():
    p = Path().resolve().parent / 'xanthydrol' / 'fits'

    sheet_path = Path().resolve().parent / 'xanthydrol' / 'measurements'
    file_name = '2025_06_11'
    dims = ['reactor','hose', 'setting', 'light', 'rep']
    common_dims = {'lamp': 'UV185'}
    generate_measurement_spreadsheet(sheet_path, file_name, dims, common_dims, make_measurement_directory=False, dilution=False)
    return


if __name__ == '__main__':
    main()