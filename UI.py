import tkinter as tk
from tkinter import messagebox
from text_classification_modified_Hoang import solve_math_problem, text_to_speech
import os

# Đường dẫn tới thư mục
audio_dir = "audio_files/"
video_dir = "video_files/"
final_dir = "final_videos/"

# Tạo thư mục nếu chưa tồn tại
for directory in [audio_dir, video_dir, final_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_text():
    # Lấy văn bản từ Textbox
    text = input_textbox.get("1.0", tk.END).strip()
    
    if not text:
        messagebox.showerror("Error", "Please enter a math problem text.")
        return

    try:
        # Giải bài toán và tạo nội dung video
        math_solution = solve_math_problem(text)
        input_file_name = "problem"
        audio_file = text_to_speech(math_solution, audio_dir + input_file_name + '.mp3')
        
        # Tạo video bằng cách gọi Manim từ dòng lệnh
        manim_command = f'manim -pql text_classification_modified_Hoang.py MathProblem -o {video_dir}{input_file_name}'
        print("Running command:", manim_command)
        os.system(manim_command)

        # Đường dẫn video được xuất từ Manim
        video_file = video_dir + input_file_name + '.mp4'
        
        if not os.path.exists(video_file):
            raise FileNotFoundError("Video file not found.")

        # Kết hợp video và âm thanh bằng FFmpeg
        final_video = final_dir + input_file_name + '.mp4'
        ffmpeg_command = f'ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {final_video}'
        print("Running FFmpeg command:", ffmpeg_command)  # Debugging output
        os.system(ffmpeg_command)

        if not os.path.exists(final_video):
            raise FileNotFoundError("Final video file not created.")

        messagebox.showinfo("Success", f"Video created: {final_video}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Math Problem to Video Converter")

# Tạo label và textbox cho input
tk.Label(root, text="Enter Math Problem:").pack(pady=5)
input_textbox = tk.Text(root, height=10, width=50)
input_textbox.pack(pady=5)

# Tạo nút để xử lý
process_button = tk.Button(root, text="Convert to Video", command=process_text)
process_button.pack(pady=20)

# Chạy ứng dụng
root.mainloop()
