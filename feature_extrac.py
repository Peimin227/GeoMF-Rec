

#!/usr/bin/env python3
import json
import pandas as pd

def load_json_lines(path, fields):
    """
    Load specified fields from a JSON lines file into a DataFrame.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            record = {field: obj.get(field, None) for field in fields}
            data.append(record)
    return pd.DataFrame(data)

def main():
    # File paths
    user_path = '/Users/peimin/Desktop/GeoMF-Rec/data/raw/yelp_academic_dataset_checkin.json'
    business_path = '/Users/peimin/Desktop/GeoMF-Rec/data/raw/yelp_academic_dataset_business.json'
    review_path = '/Users/peimin/Desktop/GeoMF-Rec/data/raw/yelp_academic_dataset_review.json'
    checkin_path = '/Users/peimin/Desktop/GeoMF-Rec/data/raw/yelp_academic_dataset_user.json'

    # 1. Load user features
    user_fields = ['user_id', 'review_count', 'average_stars', 'fans', 'yelping_since']
    df_user = load_json_lines(user_path, user_fields)

    # 2. Load business features
    biz_fields = ['business_id', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories']
    df_biz = load_json_lines(business_path, biz_fields)

    # 3. Extract checkin counts per business
    checkin_data = []
    with open(checkin_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            biz_id = obj.get('business_id')
            dates = obj.get('date', '')
            count = dates.count(',') + 1 if dates else 0
            checkin_data.append({
                'business_id': biz_id,
                'checkin_count': count
            })
    df_checkin = pd.DataFrame(checkin_data)

    # 4. Aggregate review interactions by user-business pair
    review_data = {}
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            u = obj.get('user_id')
            b = obj.get('business_id')
            date = obj.get('date')
            key = (u, b)
            if key not in review_data:
                review_data[key] = {
                    'user_business_review_count': 0,
                    'latest_review_date': date
                }
            review_data[key]['user_business_review_count'] += 1
            if date and date > review_data[key]['latest_review_date']:
                review_data[key]['latest_review_date'] = date
    df_review = pd.DataFrame([
        {
            'user_id': u,
            'business_id': b,
            'user_business_review_count': v['user_business_review_count'],
            'latest_review_date': v['latest_review_date']
        }
        for (u, b), v in review_data.items()
    ])

    # 5. Merge all into a single DataFrame
    df = (
        df_review
        .merge(df_user, on='user_id', how='left')
        .merge(df_biz, on='business_id', how='left')
        .merge(df_checkin, on='business_id', how='left')
    )

    # 6. Save to CSV
    output_path = 'data/processed/features_table.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Features table saved to {output_path}')

if __name__ == '__main__':
    main()