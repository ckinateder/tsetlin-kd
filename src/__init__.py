TEACHER_BASELINE_MODEL_PATH = "teacher_baseline.pkl"
STUDENT_BASELINE_MODEL_PATH = "student_baseline.pkl"
TEACHER_CHECKPOINT_PATH = "teacher_checkpoint.pkl"
DISTILLED_MODEL_PATH = "distilled.pkl"
OUTPUT_JSON_PATH = "output.json"
RESULTS_CSV_PATH = "results.csv"
ACCURACY_PNG_PATH = "accuracy.png"
TEST_TIME_PNG_PATH = "test_time.png"

DEFAULT_FOLDERPATH = "experiments"
DATASET_FOLDERPATH = "data"
ACC_TEST_TEACHER = "acc_test_teacher"
ACC_TEST_STUDENT = "acc_test_student"
ACC_TEST_DISTILLED = "acc_test_distilled"
TIME_TRAIN_TEACHER = "time_train_teacher"
TIME_TRAIN_STUDENT = "time_train_student"
TIME_TRAIN_DISTILLED = "time_train_distilled"
TIME_TEST_TEACHER = "time_test_teacher"
TIME_TEST_STUDENT = "time_test_student"
TIME_TEST_DISTILLED = "time_test_distilled"

PLOT_FIGSIZE = (8, 6)
PLOT_DPI = 300

RESULTS_COLUMNS = [ACC_TEST_TEACHER, ACC_TEST_STUDENT, ACC_TEST_DISTILLED, TIME_TRAIN_TEACHER, TIME_TRAIN_STUDENT,
                           TIME_TRAIN_DISTILLED, TIME_TEST_TEACHER, TIME_TEST_STUDENT, TIME_TEST_DISTILLED]

DISTILLED_DEFAULTS = {
    "teacher_num_clauses": 400,
    "student_num_clauses": 200,
    "T": 10,
    "s": 5,
    "teacher_epochs": 60,
    "student_epochs": 60,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
    "temperature": 4.0
}

DISTILLED_DEFAULTS_NESTED = {
    "teacher": {
        "C": 1000,
        "T": 10,
        "s": 5,
        "epochs": 30,
    },
    "student": {
        "C": 100,
        "T": 10,
        "s": 5,
        "epochs": 60,
    },
    "temperature": 4.0,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
}

DOWNSAMPLE_DEFAULTS = [0.05, 0.10, 0.15, 0.20, 0.25]