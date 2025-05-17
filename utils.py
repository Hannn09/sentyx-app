import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import io
import base64

# Setting Pagination Range 
def get_pagination_range(current_page, total_pages, delta=2):
    range_with_ellipsis = []
    left = current_page - delta
    right = current_page + delta + 1

    for p in range(1, total_pages + 1):
        if p == 1 or p == total_pages or (left <= p < right):
            range_with_ellipsis.append(p)
        elif range_with_ellipsis[-1] != '...':
            range_with_ellipsis.append('...')

    return range_with_ellipsis


# Generate image confusion matrix
def generate_confusion_matrix(y_true, y_pred, labels=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False, display_labels=labels)
    disp.ax_.set_title("Confusion Matrix")
    disp.ax_.set_xlabel("Predicted Label")
    disp.ax_.set_ylabel("True Label")
    plt.tight_layout()


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    
    return image_base64