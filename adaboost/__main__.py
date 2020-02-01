import sys
from adaboost import application, application_plot
from adaboost.common.check import check_application
from adaboost.common.get import application_helper

if __name__ == "__main__":

  # ==================
  # Application Helper
  # ==================

  # Print application help details
  application_help = check_application.application_help_check()
  if application_help is True:
    print("Running: Application_Helper...\n")
    app_help_status = application_helper.application_help_details()

    if app_help_status == 0:
      print("\nApplication Help Exited Successfully.")
      sys.exit(0)
    else:
      print("\nExiting. (APP_HELP_ERR)")
      sys.exit(-1)


  # ====================
  # Application Plotting
  # ====================

  # Run application with graphical representation of performance and console output
  plot_application =  check_application.plot_application_check()
  if plot_application is True:
    print("Running: Application_Plot...\n")
    app_plot_status = application_plot.run()

    if app_plot_status == 0:
      print("\nApplication Plot Exited Successfully.")
      sys.exit(0)
    else:
      print("Application: APP_PLOT_ERR")
      sys.exit(-1)


  # ===================
  # Application Default
  # ===================

  # Run default application if above not specified
  # Only accuracy results only outputted to console
  print("Running: Application_Default...\n")
  app_status = application.run()

  if app_status == 0:
    print("\nApplication Default Exited Successfully.")
    sys.exit(0)
  else:
    print("Application: APP_ERR")
    sys.exit(-1)
