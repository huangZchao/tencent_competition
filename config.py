TRAIN_DIR = '/home/baode/data1/huangzc/tencent/data/train_preliminary/'
TEST_DIR = '/home/baode/data1/huangzc/tencent/data/test/'

USER_LOG_PATH = 'grid_df_part1.pkl'
AD_INFO_PATH = 'grid_df_part2.pkl'

CLK_PATH_DICT = {'creative_id': 'clk_list.pkl',
            'advertiser_id': 'clk_list_advertiser.pkl',
            'ad_id': 'clk_list_ad.pkl',
            'product_id': 'clk_list_product.pkl',
            'product_category': 'clk_list_product_cat.pkl',
            'industry': 'clk_list_industry.pkl',
            'time': 'clk_list_time.pkl',
                 
            'per_day_click': 'per_day_click.pkl',
            'seq_statistic': 'seq_statistic.pkl',

            'creative_id_cntv_user': 'creative_id_cntv_user.npz',
            'ad_id_cntv_user': 'ad_id_cntv_user.npz',
            'advertiser_id_cntv_user': 'advertiser_id_cntv_user.npz',
            'product_id_cntv_user': 'product_id_cntv_user.npz',
            'product_category_cntv_user': 'product_category_cntv_user.npz',
            'industry_cntv_user': 'industry_cntv_user.npz',
            'time_cntv_user': 'time_cntv_user.npz',
                        
            'tfidf_creative_id': 'tfidf_creative_id.npz',
            'tfidf_ad_id': 'tfidf_ad_id.npz',
            'tfidf_product_id': 'tfidf_product_id.npz',
            'tfidf_product_category': 'tfidf_product_category.npz',
            'tfidf_advertiser_id': 'tfidf_advertiser_id.npz',
            'tfidf_industry': 'tfidf_industry.npz',
            'tfidf_time': 'tfidf_time.npz',
                 
            'gensim_ad_id': 'gensim_ad_id_dict.js',
                 
            'tfidf_stack_age': 'tfidf_stack_age.pkl',
            'tfidf_stack_gender': 'tfidf_stack_gender.pkl', 
                 
            'kfold_te_age': 'kfold_te_age.pkl',
            'kfold_te_gender': 'kfold_te_gender.pkl', 
                 
            'gensim_ad_dict': 'gensim_dict.npy'}


SUBMISSION_AGE_PATH = '/home/baode/data1/huangzc/tencent/submission/submission_age.csv'
SUBMISSION_GENDER_PATH = '/home/baode/data1/huangzc/tencent/submission/submission_gender.csv'