import csv

def generate_submision_file(X, y):
    submission_list = [['PassengerId', 'Survived']]
    for id, prediction in zip(X['PassengerId'].to_array(), y):
        submission_list.append([id, prediction])

    with open('../submission.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(submission_list)
    