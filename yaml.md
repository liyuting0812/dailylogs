## YAML学习

> 我准备用yaml文件来配置文件，取代反复修改参数的操作，使其更加便利。
>
> 开始学：

#### 1. 基本语法

> * 大小写敏感
> * 使用缩进表示层级关系
> * 使用空格space缩进，不能用Tab
> * #表注释

#### 2. YAML对象

##### （1）对象键值

> 类似字典结构：
>
> ```yaml
> key:{key1: value1, key2: value2, ...}
> #可以写成
> 
> key: 
> 	key1: value1
> 	key2: value2
> 	……
> ```



##### (2) 数组

> 以 - 开头的行表示构成一个数组。
>
> ```yaml
> companies: [{id: 1,name: company1,price: 200W},{id: 2,name: company2,price: 500W}]
> # 可以写成下面
> 
> companies:
> 	-
> 		id: 1
> 		name: company1
> 		price: 200W
> 	-
> 		id: 2
> 		name: company2
> 		price: 500W
> 
> ```

##### (3) 复合结构

> 数组和对象构成复合结构：
>
> ```json
> { 
>   languages: [ 'Ruby', 'Perl', 'Python'],
>   websites: {
>     YAML: 'yaml.org',
>     Ruby: 'ruby-lang.org',
>     Python: 'python.org',
>     Perl: 'use.perl.org' 
>   } 
> }
> ```
>
> 写成下面：
>
> ```yaml
> languages:
> 	- Ruby
> 	- Perl
> 	- Python
> websites:
> 	YAML: yaml.org
> 	Ruby: ruby-lang.org
> 	Python: python.org
> 	Perl: use.perl.org
> ```

##### (4) 纯量

> 纯量是最基本且不可再分的值：
>
> 包括**字符串、布尔值、整数、浮点数、Null、时间、日期…**
>
> ```yaml
> boolean:
> 	- True/TRUE/true
> 	- False/FALSE/false
> float：
> 	- 3.14
>     - 6.88e+5  #可以使用科学计数法
> date: 
> 	- 2000-03-03 
> 	#日期使用yyyy-MM-dd (国际标准化组织的国际标准ISO 8601是日期和时间的表示方法，全称为《数据存储和交换形式·信息交换·日期和时间的表示方法》)
> datetime: 
> 	- 2000-03-04T13:09:32+09:00 #时间与日期之间使用T连接，最后+时区
> 
> ```

##### (5) 引用

> &锚点与*别名，可以用来引用：
>
> ```yaml
> defaults:
> 	adapter: postgres
> 	host:    localhost
> 	
> development:
> 	database: myapp_development
> 	adapter:  postgres
> 	host:     localhost
> 
> test:
> 	datanase: myapp_test
> 	adapter:  postgres
>     host:     localhost
> ```
>
> 可以使用引用化简：
>
> ```yaml
> defaults: &defaults
> 	adapter: postgres
> 	host:    localhost
> 	
> development:
> 	database: myapp_development
> 	<<: *defaults
> 	
> test:
> 	database: myapp_test
> 	<<: *defaults
> ```
>
> ```yaml
> - &show Fall
> - Clark
> - *show
> ```
>
> 

