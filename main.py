import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,t


# title for the page
st.title('Decoding Confidence Intervals')

# parameters input in the sidebar
st.sidebar.title('Parameters Menu')

# mean for the population
pop_mean = st.sidebar.slider(label='Population Mean', min_value=30,
                             max_value=70,value=50,
                             step=5)


# enter the population std
pop_std = st.sidebar.slider(label='Population Std',min_value=3,
                            max_value=10,step=1)

# Number of iterations
iter_no = st.sidebar.number_input(label='Number of Iterations',min_value=10,
                                    max_value=500,value=100,step=1)

# confidence level
confidence_level = st.sidebar.slider(label='Confidence Level(in %)',
                                        min_value=5,max_value=95,
                                        value=95)

# sample size
sample_size = st.sidebar.number_input(label='Sample Size',min_value=5,
                                        max_value=500,value=50,step=5)

# generate a normal distribution
normal_distribution = np.random.normal(loc=pop_mean,scale=pop_std,
                                        size=(iter_no,sample_size)) 
# Select the test
test_name = st.sidebar.selectbox(label='Test Procedure',
                                   options=['Z Procedure','T Procedure'],
                                   index=0)


if test_name == 'Z Procedure':
    
    # take the mean for each sample
    sample_means = np.mean(normal_distribution,axis=1)
    
    # calculate the point estimate(mean of sampling distribution of sample means)
    point_estimate = np.mean(sample_means)
    
    # calculate the standard deviation of sample means
    sample_means_std = np.std(sample_means)

    # calculate the margin of error
    # critical value
    confidence_level = confidence_level/100
    z_critical = norm.ppf(confidence_level + ((1-confidence_level)/2))
    
    # standard error 
    standard_error = pop_std/np.sqrt(sample_size) 
    
    margin_of_error = z_critical * standard_error
    
    # calculate the upper bound of the interval
    upper_bound = sample_means + margin_of_error
    
    # calculate the lower bound of the interval
    lower_bound = sample_means - margin_of_error
    
    # colors chart
    colors = np.where((upper_bound > pop_mean) & (lower_bound < pop_mean),
                      'blue','orange')
    
    # apply button for confirmation:
    button = st.sidebar.button(label='Apply',key='z_apply')
    
    if button:
        # details about the test
        st.write(f'The population mean is {pop_mean} and std is {pop_std}')
        st.write(f'The critical value for {confidence_level*100}% confidence level is {z_critical:.3f}')
        st.write(f'The population std estimate is {sample_means_std * np.sqrt(sample_size):.2f}')
        st.write(f'The point estimate is {point_estimate:.2f}')
        st.write(f'The standard error is {standard_error}')
    
        # plot the figure
        fig, ax = plt.subplots(figsize=(15,6))
        ax.vlines(range(1,iter_no+1),lower_bound,upper_bound,colors=colors,label='Confidence Intervals')
        ax.axhline(pop_mean,linestyle='dashed',color='green',label='Population Mean')
        ax.scatter(range(1,iter_no+1),lower_bound,c=colors,marker='_')
        ax.scatter(range(1,iter_no+1),upper_bound,c=colors,marker='_')
        ax.scatter(range(1,iter_no+1),sample_means,c='red',label='Sample Means',s=15)
        ax.set_xlabel('No. of Iterations')
        ax.set_ylabel('Mean')
        ax.legend()
        
        st.pyplot(fig)
        
        # calculate the percentage of intervals having the population mean
        intervals_percentage = np.where(colors == 'blue',1,0)
        st.write(f'{np.sum(intervals_percentage)} out of {iter_no} confidence intervals have the population parameter')
        st.write(f'The percentage of intervals having population parameter is {(np.sum(intervals_percentage)/iter_no)*100:.2f}% with confidence level of {confidence_level*100}%')
        
        
        # generate a dataframe of the results
        df_results = pd.DataFrame()
        df_results['Sample_Mean'] = sample_means
        df_results['Upper_Bound'] = upper_bound
        df_results['Lower_Bound'] = lower_bound
        df_results['Width_of_Margin'] = upper_bound - lower_bound
        df_results['Contains_Parameter'] = intervals_percentage
        
        # replace the contains parameter column with yes and no
        df_results['Contains_Parameter'].replace({1:'Yes',0:'No'},inplace=True)
        
        # set the index starting from 1
        df_results.set_index(np.arange(1,iter_no+1),inplace=True)
        
        # show the dataframe
        st.dataframe(data=df_results)
        
    
elif test_name == 'T Procedure':
    
    # take the mean for each sample
    sample_means = np.mean(normal_distribution,axis=1)
    
    # calculate the standard deviation from each sample
    sample_stds = np.std(normal_distribution,axis=1)
    
    # calculate the point estimate(mean of sampling distribution of sample means)
    point_estimate = np.mean(sample_means)
    
    # calculate the margin of error
    
    # critical value
    confidence_level = confidence_level/100
    dof = sample_size - 1
    
    t_critical = t(df=dof).ppf(confidence_level + ((1-confidence_level)/2))
    
    # standard error 
    standard_error = np.mean(sample_stds)/np.sqrt(sample_size) 
    
    margin_of_error = t_critical * standard_error
    
    # calculate the upper bound of the interval
    upper_bound = sample_means + margin_of_error
    
    # calculate the lower bound of the interval
    lower_bound = sample_means - margin_of_error
    
    # colors chart
    colors = np.where((upper_bound > pop_mean) & (lower_bound < pop_mean),
                    'blue','orange')
    
    # apply button for confirmation:
    button = st.sidebar.button(label='Apply',key='t_apply')
    
    if button:
        # details about the test
        st.write(f'The population mean is {pop_mean} and std is {pop_std}')
        st.write(f'The critical value for {confidence_level*100}% confidence level is {t_critical:.3f}')
        st.write(f'The population std estimate is {standard_error * np.sqrt(sample_size):.2f}')
        st.write(f'The point estimate is {point_estimate:.2f}')
        st.write(f'The standard error is {standard_error}')
      
        # plot the figure
        fig, ax = plt.subplots(figsize=(15,6))
        ax.vlines(range(1,iter_no+1),lower_bound,upper_bound,colors=colors,label='Confidence Intervals')
        ax.axhline(pop_mean,linestyle='dashed',color='green',label='Population Mean')
        ax.scatter(range(1,iter_no+1),lower_bound,c=colors,marker='_')
        ax.scatter(range(1,iter_no+1),upper_bound,c=colors,marker='_')
        ax.scatter(range(1,iter_no+1),sample_means,c='red',label='Sample Means',s=15)
        ax.set_xlabel('No. of Iterations')
        ax.set_ylabel('Mean')
        ax.legend()
        
        st.pyplot(fig)
        
        # calculate the percentage of intervals having the population mean
        intervals_percentage = np.where(colors == 'blue',1,0)
        st.write(f'{np.sum(intervals_percentage)} out of {iter_no} confidence intervals have the population parameter')
        st.write(f'The percentage of intervals having population parameter is {(np.sum(intervals_percentage)/iter_no)*100:.2f}% with confidence level of {confidence_level*100}%')
        
        
        # generate a dataframe of the results
        df_results = pd.DataFrame()
        df_results['Sample_Mean'] = sample_means
        df_results['Upper_Bound'] = upper_bound
        df_results['Lower_Bound'] = lower_bound
        df_results['Width_of_Margin'] = upper_bound - lower_bound
        df_results['Contains_Parameter'] = intervals_percentage
        
        # replace the contains parameter column with yes and no
        df_results['Contains_Parameter'].replace({1:'Yes',0:'No'},inplace=True)
        
        # set the index starting from 1
        df_results.set_index(np.arange(1,iter_no+1),inplace=True)
        
        # show the dataframe
        st.dataframe(data=df_results)