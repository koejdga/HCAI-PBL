from django import forms


class CSVUploadForm(forms.Form):
    file = forms.FileField(label="Select a CSV file")

    def clean_file(self):
        uploaded_file = self.cleaned_data["file"]
        if not uploaded_file.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a CSV file.")
        return uploaded_file
