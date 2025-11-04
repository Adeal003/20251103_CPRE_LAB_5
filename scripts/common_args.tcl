set option {
  {xsa_path.arg       ""                        "path to XSA file"}
  {workspace_dir.arg  "workspace"               "workspace directory"}
  {framework_dir.arg  "."                       "path to framework repository"}
  {source_dir.arg     "./src"                   "path to src directory in framework repository"}
  {project_name.arg   "mlp"                     "application project name"}
  {platform_name.arg	""		                	  "platform project name"}
  {hwproject_name.arg ""                        "hardware project name"}
  {proc_name.arg      "ps7_cortexa9_0"          "prorcessor name"}
  {template.arg       "Empty Application (C++)" "application project template"}
  {prepare_debug.arg  "no"                      "(flash) prepare for debugging instead of flashing"}
}

set usage "<arguments>"
array set params [cmdline::getoptions argv $option $usage]

set params(xsa_path) [file normalize $params(xsa_path)]
set params(workspace_dir) [file normalize $params(workspace_dir)]
set params(framework_dir) [file normalize $params(framework_dir)]
set params(source_dir) [file normalize $params(source_dir)]

if {$params(platform_name) eq ""} {
  set params(platform_name) "$params(project_name)_platform"
}

if {$params(hwproject_name) eq ""} {
  set params(hwproject_name) "$params(project_name)_hwproj"
}
