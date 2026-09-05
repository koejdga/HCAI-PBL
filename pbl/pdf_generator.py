import os
from django.conf import settings
from django.template.loader import get_template
from django.http import HttpResponse
from xhtml2pdf import pisa

def link_callback(uri, rel):
    """
    Convert HTML images, fonts, and stylesheets links to local files on disk.
    This is required by xhtml2pdf to load assets locally in the Django framework.
    """
    # 1. Resolve static files
    if uri.startswith(settings.STATIC_URL):
        # First try STATIC_ROOT
        path = os.path.join(settings.STATIC_ROOT, uri.replace(settings.STATIC_URL, ""))
        if not os.path.exists(path):
            # Fallback to directories in STATICFILES_DIRS
            for static_dir in settings.STATICFILES_DIRS:
                trial_path = os.path.join(static_dir, uri.replace(settings.STATIC_URL, ""))
                if os.path.exists(trial_path):
                    path = trial_path
                    break
        return path
        
    # 2. Resolve media files
    elif uri.startswith(settings.MEDIA_URL):
        path = os.path.join(settings.MEDIA_ROOT, uri.replace(settings.MEDIA_URL, ""))
        return path
        
    # 3. Return unchanged uri for external links or absolute paths
    return uri

def render_to_pdf(template_src, context_dict={}, filename="report.pdf"):
    """
    Renders a Django HTML template to a PDF HttpResponse using xhtml2pdf.
    """
    template = get_template(template_src)
    html = template.render(context_dict)
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    # Generate PDF
    pisa_status = pisa.CreatePDF(
        html,
        dest=response,
        link_callback=link_callback,
        encoding='utf-8'
    )
    
    if pisa_status.err:
        return HttpResponse("Error generating PDF report", status=500)
        
    return response
