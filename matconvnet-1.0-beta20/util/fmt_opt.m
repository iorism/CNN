function str = fmt_opt(name, opt)

tmpl_opt = repmat('%d ',1, numel(opt));
tmpl = sprintf('%s: [%s]', name, tmpl_opt);
str = sprintf(tmpl, opt);

