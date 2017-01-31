import os.path
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

class DataLoader:

    MRI_FOLDER = 'MRI' # MRI folder name
    MRI_DATASET = 'img_array_train_6k_' # MRI dataset to test
    MRI_VALIDSET = 'img_array_valid_6k_' # MRI dataset to test
    MRI_TESTSET = 'img_array_test_6k_' # MRI dataset to test
    MRI_DATASET_EXT = '.npy'
    IMAGES_PER_SCAN = 62
    IMAGES_DIMENSION = 96
    DEMO_MASTER = 'adni_demographic_master_kaggle.csv' # CSV Base lookup table
    FIGURE_FOLDER = 'figures'
    RUN_FOLDER = 'run'
    SAVE_FIGURE = True
        
    @staticmethod
    def load_table(os_path='media/user/'):
        """ Loading csv lookup table ocated on usb key 
        File adni_demographic_master_kaggle.csv contains multiples fields.
            * train_valid_test : Field interpreted as Training 0, Validation 1, Test 2
            * only_date : MRI scan date, as format %Y%m%d (Year, month, day, e.g.: 20051021)
            * diagnosis : Field interpreted as : Normal 1, MCI (Mild Cognitive Impairment) 2, AD (Alzheimer's Disease) 3
            * sex : Field interpreted as : Female = 'F', Male = 'M'
            * age_at_scan : Age of the patient when MRI scan was performed
        """
        # Build csv fiename
        csv_filename = os.path.join(os_path, DataLoader.DEMO_MASTER)
        # Load CSV file to process subject data
        try:
            return pd.read_csv(csv_filename)
        except OSError:
            # Handle eror in case of csv_filename not found
            print("ERROR - LOAD_FILE - No file", csv_filename, "found")
            return
               
    @staticmethod
    def load_MRI_dataset_part(os_path='media/user/', nbr=1):
        """ Load part of dataset to display simple preview. nbr indicate the amount of packages
        """
        # Check if nbr arg is positive and bigger than one
        if nbr < 1:
            print("ERROR - LOAD_MRI_DATASET - Bad argument nbr", nbr)
            return
        
        try:
            # Build filename and load first file
            strFileName = DataLoader.MRI_DATASET + str(1) + DataLoader.MRI_DATASET_EXT
            pathMRI = os.path.join(os_path, DataLoader.MRI_FOLDER, strFileName)
            print('Loading...', pathMRI)
            data = np.load(pathMRI)
            # Do same over all dataset with nmax = nbr
            for i in range(nbr-1):
                strFileName = DataLoader.MRI_DATASET + str(i+2) + DataLoader.MRI_DATASET_EXT
                pathMRI = os.path.join(os_path, DataLoader.MRI_FOLDER, strFileName)
                print('Loading...', pathMRI)
                data = np.concatenate((data, np.load(pathMRI)), axis=0)
        except OSError:            
            # Handle eror in case of csv_filename not found
            print("ERROR - LOAD_MRI_DATASET - No file", pathMRI, "found")
            return
        # Must gave a integer value of patients, therefore uncomplete dataframe are dropped
        Nfull = data.shape[0]//DataLoader.IMAGES_PER_SCAN
        # Compute number of dropped images
        Ndropped = data.shape[0] - Nfull*DataLoader.IMAGES_PER_SCAN
        # Return only complete data (e.g. len(data) is a multiple of IMAGES_PER_SCAN)
        data = data[0:Nfull*DataLoader.IMAGES_PER_SCAN, :, :]
        print('Data loaded with shape=', data.shape, '\nNomber patients=', Nfull, '\nImages dropped=', Ndropped)
        return data
    
    @staticmethod
    def load_MRI_dataset_part_per_patient(os_path='media/user/', nbr=1):
        """ Load part of dataset to display simple preview. nbr indicate the amount of packages. The shape is [nb of patient, nb of images (62), image dim 1 (96), image dim 2 (96)]
        """
        # Check if nbr arg is positive and bigger than one
        if nbr < 1:
            print("ERROR - LOAD_MRI_DATASET - Bad argument nbr", nbr)
            return
        
        data = DataLoader.load_MRI_dataset_part(os_path=os_path, nbr=nbr)
        data2 = np.reshape(data,[data.shape[0]//DataLoader.IMAGES_PER_SCAN, DataLoader.IMAGES_PER_SCAN,
                                 DataLoader.IMAGES_DIMENSION, DataLoader.IMAGES_DIMENSION])
        return data2
            
    @staticmethod
    def load_MRI_dataset_part_id(os_path='media/user/', nbr=1, id_image = [32], set_type='train', merged=True):
        """ Load part of dataset to display simple preview. nbr indicate the amount of packages. Only one image is kept per set.
            The value of id_image indicate the image kept. 0 <= id_image < IMAGES_PER_SCAN
        """

        # Check if nbr arg is positive and bigger than one
        if nbr < 1:
            print("ERROR - load_MRI_dataset_part_mid - Bad argument nbr", nbr)
            return

        # Keep trace of datalength to compute correct index references
        len_data_tot = 0
        try:
            # Take range 0 to IMAGES_PER_SCAN. Then substrect id_image. Therefore interval will be : 
            # -id_image to IMAGES_PER_SCAN-id_image. Every multiple of IMAGES_PER_SCAN will be the images to keep. Index is the
            # location of nonzeros after modulo operation.

            data_frame = np.zeros((0, DataLoader.IMAGES_DIMENSION, DataLoader.IMAGES_DIMENSION))

            for i in range(nbr):
                # Build path to data
                pathMRI = DataLoader.getNextPath(os_path, i+1, set_type)
                # Print progression
                print('Loading...', pathMRI, ' ... current images=', data_frame.shape[0])
                data = np.load(pathMRI)
                len_data = len(data)

                id_all = np.zeros(0)
                for j, val_id in enumerate(id_image):
                    # Same index operation as before. In this case len_data_tot must to be take into acount
                    ids = np.nonzero(np.mod(
                            np.linspace(0,len_data-1,len_data) - id_image[j] + len_data_tot, 
                            DataLoader.IMAGES_PER_SCAN) == 0)[0]
                    id_all = np.append(id_all, ids)
                data_frame = np.concatenate((data_frame, data[np.sort(id_all.astype(int))]), axis=0)
                # Set new data total length
                len_data_tot += len_data

        except OSError:
            # Handle eror in case of csv_filename not found
            print("ERROR - LOAD_MRI_DATASET - No file", pathMRI, "found")
            return

        # Reshape to get corresponding size (multiple of id_image)
        n_patient_full = data_frame.shape[0]//len(id_image)
        data_frame = data_frame[:n_patient_full*len(id_image)]
        data_frame = np.reshape(data_frame, 
                                [n_patient_full, len(id_image), DataLoader.IMAGES_DIMENSION, DataLoader.IMAGES_DIMENSION])
        if merged:
            # Merge image as a single on
            data_frame = np.reshape(data_frame, 
                                [n_patient_full, len(id_image)*DataLoader.IMAGES_DIMENSION, DataLoader.IMAGES_DIMENSION])

        return data_frame
    
    @staticmethod
    def getNextPath(os_path='media/user/', index=1, set_type='train'):
        """ Build path to MRI file off the form : path/to/mri/MRI_DATASET+index+MRI_DATASET_EXT
        """
        str_dataset = DataLoader.MRI_DATASET
        if set_type is 'valid':
            str_dataset = DataLoader.MRI_VALIDSET
        elif set_type is 'test':
            str_dataset = DataLoader.MRI_TESTSET
            
        strFileName = str_dataset + str(index) + DataLoader.MRI_DATASET_EXT
        return os.path.join(os_path, DataLoader.MRI_FOLDER, strFileName)
        
    @staticmethod
    def clean_data(patients=None):
        """ Cleaning data 
            * train_valid_test : int to category (TRAINING, VALIDATION, TEST)
            * only_date : str to date (%Y%m%d)
            * diagnosis : int to category (NORMAL, MCI, AD)
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - CLEAN_DATA - patients is None')
            return
        # Set catergory for train_valid_test as : Training = 0, Validation = 1, Test = 2
        patients['train_valid_test'] = patients['train_valid_test'].astype('category')
        patients['train_valid_test'].cat.categories = ['TRAINING', 'VALIDATION', 'TEST']
        # Convert to date format
        patients['only_date'] = pd.to_datetime(patients['only_date'], format='%Y%m%d', errors='ignore')
        # Set catergory for cognitively as : Normal = 1, MCI = 2, AD = 3
        patients['diagnosis'] = patients['diagnosis'].astype('category')
        patients['diagnosis'].cat.categories = ['NORMAL', 'MCI', 'AD']
        return patients
        
    @staticmethod
    def print_correlation(patients=None):
        """ Print correlation between diagnosis and age/sex
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - PRINT_HEAD - patients is None')
            return
        # Convert 'F'=1 and 'M'=0 for genders
        patient_sex_num = patients['sex'].replace({'F': 1, 'M': 2})
        # Get categorie and set values 'NORMAL'=1, 'MCI'=2, 'AD'=3
        patient_diag_num = patients['diagnosis'].copy()
        patient_diag_num.cat.categories = [1, 2, 3]
        
        # Get correlation between AD and age
        corr_age = patients['age_at_scan'].corr(patient_diag_num, method='pearson')
        # Get correlation between AD and sex
        corr_sex = patient_sex_num.corr(patient_diag_num, method='pearson')
        
        # Print results
        strCorr = 'Correlation Diagnosis - Age: ' + str(np.round(corr_age, 2))
        strCorr += '\nCorrelation Diagnosis - Sex: ' + str(np.round(corr_sex, 2))
        print(strCorr)
    
    @staticmethod
    def plot_pie_info(patients):
        """ Dispaly pie chart info about 
            * Train/Validation/Test repartition
            * Diagnosis repartition in percentage
            * Genre repartition in percentage
        """        
        if patients is None:
            print('ERROR - PLOT_PIE_INFO - patients is None')
            return
        
        # REMOVE below line
        # print( patients['train_valid_test'].value_counts())
        
        # Display data repartition (overall)
        fig = plt.figure(figsize=(12,4));
        plt.subplot(1,3,1)
        # Diagnosis repartition pie chart
        plt.pie(patients['train_valid_test'].value_counts(), explode=(0.1, 0.1, 0.1), labels=patients['train_valid_test'].cat.categories,
                autopct='%1.1f%%', startangle=0);
        plt.title('Train/Valid/Test', fontsize=18 );
        plt.subplot(1,3,2)
        # Diagnosis repartition pie chart
        plt.pie(patients['diagnosis'].value_counts(), explode=(0.1, 0.1, 0.1), labels=patients['diagnosis'].cat.categories,
                autopct='%1.1f%%', startangle=90);
        plt.title('Diagnosis', fontsize=18 );
        plt.subplot(1,3,3)
        # Sex (gender) repartition pie chart
        plt.pie(patients['sex'].value_counts(), explode=(0.1, 0.1), labels=['Female', 'Male'],
                autopct='%1.1f%%', startangle=90);
        plt.title('Gender', fontsize=18); plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'pie_diag_gender.pdf')
        
    @staticmethod
    def plot_age_distrib(patients=None):
        """ Display age distribution for all 3 diagnosis cases (Normal, MCI, AD). Since the data does not have a 
            normal distribution the stddev is nor relevant. It will only indicate how 'speard' are the data 
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - PLOT_AGE_DISTRIB - patients is None')
            return
        # Get min/max range to display plots
        xMin = np.min(patients['age_at_scan']); xMax = np.max(patients['age_at_scan']);
        # Get basic data repartition features (not gaussian, but nformation about spread)
        patients_normal = patients[patients['diagnosis']=='NORMAL']['age_at_scan']
        patients_mci = patients[patients['diagnosis']=='MCI']['age_at_scan']
        patients_ad = patients[patients['diagnosis']=='AD']['age_at_scan']
        mean_normal = np.mean(patients_normal); stddev_normal = np.std(patients_normal)
        mean_mci = np.mean(patients_mci); stddev_mci = np.std(patients_mci)
        mean_ad = np.mean(patients_ad); stddev_ad = np.std(patients_ad)

        # Get for each diagnosis spectific value for men and female. Data will appear as stacked bars
        patients_normal_f = patients[np.logical_and(patients['diagnosis']=='NORMAL', patients['sex']=='F')]['age_at_scan']
        patients_normal_m = patients[np.logical_and(patients['diagnosis']=='NORMAL', patients['sex']=='M')]['age_at_scan']
        patients_mci_f = patients[np.logical_and(patients['diagnosis']=='MCI', patients['sex']=='F')]['age_at_scan']
        patients_mci_m = patients[np.logical_and(patients['diagnosis']=='MCI', patients['sex']=='M')]['age_at_scan']
        patients_ad_f = patients[np.logical_and(patients['diagnosis']=='AD', patients['sex']=='F')]['age_at_scan']
        patients_ad_m = patients[np.logical_and(patients['diagnosis']=='AD', patients['sex']=='M')]['age_at_scan']

        # The histogram of the data for all patients (one per case)
        width = 16; height = 4; sep = 20;
        fig = plt.figure(figsize=(width, height))
        # Plot diagnosis - Normal
        plt.subplot(1,3,1)
        # Plot stacked histogram female and male
        plt.hist( [patients_normal_f, patients_normal_m], sep, stacked=True); plt.legend(['Female','Male'], loc=2); plt.grid();
        plt.xlabel('Age'); plt.ylabel('Number of Scan'); plt.xlim([xMin, xMax]);
        # Set title with mean vaue and variance
        plt.title('NORMAL\n $\mu$='+str(np.round(mean_normal,1)) 
                  + ' / $\sigma$=' + str(np.round(stddev_normal,1)), fontsize=16); 
        plt.ylim([0,160])
        # Plot diagnosis - MCI
        plt.subplot(1,3,2)
        plt.hist([patients_mci_f, patients_mci_m], sep, stacked=True); plt.legend(['Female','Male'], loc=2); plt.grid();
        plt.xlabel('Age'); plt.ylabel('Number of Scan'); plt.xlim([xMin, xMax]);
        plt.title('Mild Cognitive Impairment\n $\mu$='+str(np.round(mean_mci,1)) 
                  + ' / $\sigma$=' + str(np.round(stddev_mci,1)), fontsize=16);
        plt.ylim([0,160])
        # Plot diagnosis - AD
        plt.subplot(1,3,3)
        plt.hist([patients_ad_f, patients_ad_m], sep, stacked=True); plt.legend(['Female','Male']); plt.grid();
        plt.xlabel('Age'); plt.ylabel('Number of Scan'); plt.xlim([xMin, xMax]);
        plt.title('Alzheimer\'s Disease\n $\mu$='+str(np.round(mean_ad,1)) 
                  + ' / $\sigma$=' + str(np.round(stddev_ad,1)), fontsize=16);
        plt.ylim([0,160])
        plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_age_distrib.pdf')

    @staticmethod
    def plot_scan_time_hist(patients=None):
        """ Display time distribution of scans.
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - PLOT_SCAN_TIME_HIST - patients is None')
            return
        
        # Plot histogram of scans
        fig = plt.figure(figsize=(16,4))
        patients['only_date'].hist(bins=40)
        plt.title('Scan - Time repartition', fontsize=18); plt.xlabel('Year'); plt.ylabel('Amount')
        plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_scan_time_hist.pdf')
            
    @staticmethod
    def plot_scan_time_age(patients=None):
        """ Display age of patient over time and color indicationg gender
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - PLOT_SCAN_TIME_AGE - patients is None')
            return
        # Create linear model of data to estimate how measure were performed over time (patient age)
        model = linear_model.LinearRegression()
        # Fit data over time
        model.fit(patients['only_date'].reshape(-1, 1), patients['age_at_scan'].reshape(-1, 1))
        # Get predicted (model line)
        y_pred = model.predict(patients['only_date'].values.astype('float64').reshape(-1, 1))

        # Plot histogram of scans
        fig = plt.figure(figsize=(16,4))
        # Plot female and male points over time
        plt.plot_date(patients[patients['sex']=='F']['only_date'], patients[patients['sex']=='F']['age_at_scan'], c=[0,1,0,0.5])
        plt.plot_date(patients[patients['sex']=='M']['only_date'], patients[patients['sex']=='M']['age_at_scan'], c=[1,0,0,0.5])
        # Add linear regression model 
        plt.plot(patients['only_date'], y_pred, linewidth=3); plt.grid();
        plt.xlabel('Year'); plt.ylabel('Age')
        plt.title('Scan - Patient age over time', fontsize=18)
        plt.legend(['Female','Male','Lin. Reg.'], loc=4)
        plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_scan_time_age.pdf')
    
    @staticmethod
    def plot_MRI_patients(patients=None, MRI=None, n_patient = 4, scan_ids =[0, 1, 30, 31, 50, 51, 52, 53], diagnosis='NORMAL'):
        """ Display multiples patients scan. The scan to compare are given by : scan_id. n_patient is the number of patient
            to compare. To acheive a more meaningfull comparaison, patitent with same diagnosis are selected. diagnosis can
            have value : 'NORMAL', 'MCI', 'AD'.
        """
        # Check if patient is an empty matrix
        if patients is None:
            print('ERROR - PLOT_SCAN_TIME_AGE - patients is None')
            return
        # Get number of patients (full)
        n_max = MRI.shape[0]//DataLoader.IMAGES_PER_SCAN
        # Extract ids only for patient with corresponding diagnosis
        patientNormalId = np.nonzero(patients.loc[:n_max,'diagnosis'] == diagnosis)[0]
        patientId = patientNormalId[np.random.randint(low=0, high=len(patientNormalId), size=(1,n_patient))]

        # Create extraction ids
        id_base = np.ones((n_patient, 1)).dot(np.array(scan_ids).reshape(1,-1))
        id_patient_mul = DataLoader.IMAGES_PER_SCAN * np.ones((len(scan_ids), 1)).dot(patientId).T
        selection_ids = id_patient_mul+id_base
        
        # Plot scan ids for multiple patients (to compare brain shapes)
        fig = plt.figure(figsize=(2*len(scan_ids),2*n_patient))
        # Patient axis (line)
        for i in range(n_patient):
            # Scan axis (column)
            for j in range(len(scan_ids)):
                # Plot actual image
                plt.subplot(n_patient, len(scan_ids), i*selection_ids.shape[1]+j+1)
                plt.imshow(MRI[selection_ids.astype(int)[i, j]])
                plt.axis('off')
                if i == 0:
                    plt.title('Scan ' + str(scan_ids[j]))
        plt.suptitle('Scan - Comparaison with multiple patients - ' + diagnosis, fontsize=18)
        plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_scan_patients.pdf')

    @staticmethod
    def plot_MRI_diagnosis(patients=None, MRI=None, n_patient=4, id_scan=10, scale=4):
        """ Display MRIs of each diagnosis (Normal, MCI, AD). n_patients is the number of patients to plot (x axis).
            id_scan is the scan_id thaht must be part of interval [0 IMAGES_PER_SCAN]. Scale is the image output size factor.
        """
        # Check if patient or MRI is an empty matrix
        if patients is None or MRI is None:
            print('ERROR - PLOT_MRI - patients or MRI are None')
            return
        
        # Set images per scan
        n_max = MRI.shape[0]//DataLoader.IMAGES_PER_SCAN
        ids = np.zeros((3, n_patient))

        # Get first MRIs for each cases. Since one picture is enougth, central one is taken
        for i, diagnosis in enumerate(patients['diagnosis'].cat.categories):
            ids_patient = np.nonzero(patients.loc[:n_max, 'diagnosis'] == diagnosis)[0]
            ids[i,:] = ids_patient[np.random.randint(low=0, high=len(ids_patient)-1, size=(1,n_patient))]

        ids = ids*DataLoader.IMAGES_PER_SCAN + id_scan
        fig = plt.figure(figsize=(scale*n_patient,scale*3))
        for i in range(3):
            for j in range(n_patient):
                plt.subplot(3, n_patient, n_patient*i + j + 1)
                plt.title(patients['diagnosis'].cat.categories[i])
                plt.imshow(MRI[ids.astype(int)[i,j]])
                plt.axis('off');
        plt.suptitle('Scan - Different patient/diagnosis', fontsize=18)
        plt.show()

        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_mri_preview.pdf')
            
    @staticmethod
    def save_plot(fig=None, name=None):
        """ Save plot to folder defined by DataLoader.FIGURE_FOLDER and create it if not existing
        """
        # Check if correct arguments
        if fig is None or name is None:
            print('ERROR - SAVE_PLOT - fig or name are None')
            return
        # Check if figure folder already exists
        if not os.path.isdir(DataLoader.FIGURE_FOLDER):
            print('Creating folder', DataLoader.FIGURE_FOLDER, '...')
            os.mkdir(DataLoader.FIGURE_FOLDER) 
        # Save plot to folder
        pp = PdfPages(os.path.join(DataLoader.FIGURE_FOLDER, name))
        fig.savefig(pp, format='pdf')
        pp.close()
        
    @staticmethod
    def save_run(run_dic=None, name=None):
        """ Save run to folder defined by DataLoader.RUN_FOLDER and create it if not existing
        """
        # Check if correct arguments
        if run_dic is None or name is None:
            print('ERROR - SAVE_RUN - run_dic or name are None')
            return
        # Check if figure folder already exists
        if not os.path.isdir(DataLoader.RUN_FOLDER):
            print('Creating folder', DataLoader.RUN_FOLDER, '...')
            os.mkdir(DataLoader.RUN_FOLDER) 
        # Save run to folder
        print('Saving run ', os.path.join(DataLoader.RUN_FOLDER, name), '...' )
        np.save(os.path.join(DataLoader.RUN_FOLDER, name), run_dic) 
        print('Saved')
