TRAIN_DIR = '/home/huangzc/competition/tencent/data/train_preliminary/'
TEST_DIR = '/home/huangzc/competition/tencent/data/test/'

USER_LOG_PATH = 'grid_df_part1.pkl'
AD_INFO_PATH = 'grid_df_part2.pkl'

# CLK_LIST_CREATIVE_PATH = 'clk_list.pkl'
# CLK_LIST_ADVERTISER_PATH = 'clk_list_advertiser.pkl'
# CLK_LIST_AD_PATH = 'clk_list_ad.pkl'
# CLK_LIST_PRODUCT_PATH = 'clk_list_product.pkl'
# CLK_LIST_PRODUCT_CAT_PATH = 'clk_list_product_cat.pkl'
# CLK_LIST_INDUSTRY_PATH = 'clk_list_industry.pkl'

CLK_PATH_DICT = {'creative_id': 'clk_list.pkl',
            'advertiser_id': 'clk_list_advertiser.pkl',
            'ad_id': 'clk_list_ad.pkl',
            'product_id': 'clk_list_product.pkl',
            'product_category': 'clk_list_product_cat.pkl',
            'industry': 'clk_list_industry.pkl',
            'per_day_click': 'per_day_click.pkl',
            'kfold_te': 'kfold_te.pkl',
            'seq_statistic': 'seq_statistic.pkl',
            'cross_feature': 'cross_feature.pkl',
            'tfidf_stack': 'tfidf_stack.pkl'}


SUBMISSION_AGE_PATH = '/home/huangzc/competition/tencent/submission/submission_age.csv'
SUBMISSION_GENDER_PATH = '/home/huangzc/competition/tencent/submission/submission_gender.csv'